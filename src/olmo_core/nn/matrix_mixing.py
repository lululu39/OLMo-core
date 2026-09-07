"""Static cross-layer linear combinations of trainable weight matrices."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError


@dataclass
class MatrixMixingConfig(Config):
    """
    Enable static matrix mixing for a family of transformer projections.

    Each projection kind has its own bank of ``num_bases`` matrices shared across
    all layers. Coefficients are unconstrained, trainable, and independent of the input.
    """

    num_bases: int = 4
    """Number of shared bases per projection kind (K)."""

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Reject invalid basis counts, including after configuration overrides."""
        if type(self.num_bases) is not int or self.num_bases < 1:
            raise OLMoConfigurationError("Matrix mixing num_bases must be a positive integer")


class MatrixBasisBank(nn.Module):
    """
    Own a single registered tensor of bases with shape ``(K, out_features, in_features)``.

    Banks belong to the transformer root, never to individual projections.
    """

    def __init__(self, linear: nn.Linear, num_bases: int, init_std: float):
        super().__init__()
        self.init_std = init_std
        self.bases = nn.Parameter(
            torch.empty(
                num_bases,
                linear.out_features,
                linear.in_features,
                dtype=linear.weight.dtype,
                device=linear.weight.device,
            )
        )

    @torch.no_grad()
    def init_weights(self, generator: Optional[torch.Generator] = None) -> None:
        """Randomly initialize every basis once, including when sharded by FSDP."""
        from olmo_core.nn.transformer.init import _apply_init

        _apply_init(
            nn.init.trunc_normal_,
            self.bases,
            std=self.init_std,
            a=-3 * self.init_std,
            b=3 * self.init_std,
            generator=generator,
        )


class MixedLinear(nn.Module):
    r"""
    A linear projection with :math:`W_l = \sum_k a_{l,k} B_k`.

    Only ``coefficients`` and the optional unshared bias are registered here.
    The bank reference is deliberately not registered as a child module. Looking up
    its parameter at each call preserves device moves, meta materialization, FSDP
    parameter replacement, and state-dict loading. Deepcopy preserves this reference
    within the copied model as well.

    A single matrix-vector product materializes the effective weight, followed by
    ordinary :func:`torch.nn.functional.linear`. No token routing, softmax, or
    persistent effective-weight cache is used.
    """

    def __init__(self, linear: nn.Linear, bank: MatrixBasisBank):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        object.__setattr__(self, "_bank", bank)
        self.coefficients = nn.Parameter(
            torch.empty(bank.bases.shape[0], dtype=linear.weight.dtype, device=linear.weight.device)
        )
        self.register_parameter("bias", linear.bias)

    @property
    def bank(self) -> MatrixBasisBank:
        """The shared bank, owned and registered by the transformer root."""
        return self._bank

    @property
    def weight(self) -> torch.Tensor:
        """Materialize the effective weight without caching it across calls."""
        bases = self.bank.bases
        return torch.matmul(self.coefficients.to(bases.dtype), bases.flatten(1)).view(
            self.out_features, self.in_features
        )

    @torch.no_grad()
    def init_weights(self, *, std: float, generator: Optional[torch.Generator] = None) -> None:
        """
        Initialize random coefficients with norm ``std / bank.init_std``.

        This matches the dense projection's initialization variance (including
        depth scaling), independently of K. Normalization applies only at init.
        """
        from olmo_core.nn.transformer.init import _apply_init

        def init_coefficients(tensor: torch.Tensor):
            # Normalize in float32 even for bfloat16 models.
            values = torch.randn(tensor.shape, device=tensor.device, generator=generator)
            values = F.normalize(values, dim=0) * (std / self.bank.init_std)
            tensor.copy_(values)

        _apply_init(init_coefficients, self.coefficients)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the mixed weight to an input of shape ``(*, in_features)``."""
        return F.linear(x, self.weight, self.bias)


def mix_block_projections(
    block: nn.Module,
    banks: nn.ModuleDict,
    *,
    mlp: Optional[MatrixMixingConfig],
    attn: Optional[MatrixMixingConfig],
    init_std: float,
) -> None:
    """Replace selected dense projections, reusing the root's banks across blocks."""
    from olmo_core.nn.attention import Attention
    from olmo_core.nn.feed_forward import FeedForward

    for config, module_name, module_type, names in (
        (mlp, "feed_forward", FeedForward, ("w1", "w2", "w3")),
        (attn, "attention", Attention, ("w_q", "w_k", "w_v", "w_out")),
    ):
        if config is None:
            continue
        config.validate()
        module = getattr(block, module_name, None)
        if type(module) is not module_type:
            raise OLMoConfigurationError(
                f"Matrix mixing requires standard {module_type.__name__} in every block; "
                "normalized, fused-QKV, recurrent, and MoE projections are not supported"
            )
        for name in names:
            linear = getattr(module, name)
            key = f"{module_name}_{name}"
            if key not in banks:
                banks[key] = MatrixBasisBank(linear, config.num_bases, init_std)
            bank = banks[key]
            if bank.bases.shape != (config.num_bases, *linear.weight.shape) or (
                bank.bases.dtype != linear.weight.dtype
            ):
                raise OLMoConfigurationError(
                    f"Matrix mixing requires identical shapes and dtypes across layers for {key}"
                )
            setattr(module, name, MixedLinear(linear, bank))


def linear_parameter_count(module: nn.Module) -> int:
    """Count effective dense parameters for FLOP estimates, excluding mixing overhead."""
    count = sum(p.numel() for p in module.parameters())
    for child in module.modules():
        if isinstance(child, MixedLinear):
            count += child.in_features * child.out_features - child.coefficients.numel()
    return count
