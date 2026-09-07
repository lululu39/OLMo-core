import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import init_device_mesh

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import get_full_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.matrix_mixing import MatrixBasisBank, MixedLinear
from olmo_core.nn.transformer import (
    InitMethod,
    MatrixMixingConfig,
    TransformerActivationCheckpointingMode,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.testing import requires_gpu, requires_multi_gpu, run_distributed_test
from olmo_core.utils import get_default_device


def tiny_config(mlp=True, attn=True, **kwargs):
    return TransformerConfig.olmo3_1M(
        vocab_size=64,
        n_kv_heads=2,
        hidden_size_multiple_of=8,
        attn_backend=AttentionBackendName.torch,
        mlp_matrix_mixing=MatrixMixingConfig(2) if mlp else None,
        attn_matrix_mixing=MatrixMixingConfig(3) if attn else None,
        **kwargs,
    )


@pytest.mark.parametrize("mlp,attn", [(False, False), (True, False), (False, True), (True, True)])
def test_registration_config_and_dense_equivalence(mlp, attn):
    config = tiny_config(mlp, attn)
    restored = TransformerConfig.from_dict(config.as_config_dict())
    assert restored.as_config_dict() == config.as_config_dict()
    model = restored.build(init_device="meta")
    model.init_weights(device=torch.device("cpu"))
    assert config.num_params == model.num_params == sum(p.numel() for p in model.parameters())
    assert config.num_active_params == model.num_params
    assert model.num_flops_per_token(8) == tiny_config(False, False).build().num_flops_per_token(8)
    params = list(model.named_parameters(remove_duplicate=False))
    assert len(params) == len({id(p) for _, p in params})
    state = model.state_dict()
    assert sum(n.startswith("matrix_bases.") for n in state) == 3 * mlp + 4 * attn
    mixed = [(n, m) for n, m in model.named_modules() if isinstance(m, MixedLinear)]
    assert len(mixed) == config.n_layers * (3 * mlp + 4 * attn)
    for name, projection in mixed:
        assert name + ".weight" not in state
        assert set(dict(projection.named_parameters())) == {"coefficients"}
        assert projection.bank in model.matrix_bases.values()

    # Compare the entire network to independent dense effective weights. Their
    # gradients give an independent reference for the chain rule into a and B.
    dense = copy.deepcopy(model)
    for name, projection in list(dense.named_modules()):
        if isinstance(projection, MixedLinear):
            linear = nn.Linear(projection.in_features, projection.out_features, bias=False)
            with torch.no_grad():
                linear.weight.copy_(projection.weight)
            parent, attr = name.rsplit(".", 1)
            setattr(dense.get_submodule(parent), attr, linear)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    actual, expected = model(inputs), dense(inputs)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    bank_grads: dict[int, torch.Tensor] = {}
    for name, projection in mixed:
        grad = dense.get_submodule(name).weight.grad
        bases = projection.bank.bases
        torch.testing.assert_close(
            projection.coefficients.grad, (bases.detach() * grad).sum(dim=(1, 2))
        )
        contribution = projection.coefficients.detach()[:, None, None] * grad
        bank_grads.setdefault(id(bases), torch.zeros_like(bases)).add_(contribution)
    for _, projection in mixed:
        torch.testing.assert_close(
            projection.bank.bases.grad, bank_grads[id(projection.bank.bases)]
        )


@pytest.mark.parametrize("num_bases", [1, 2, 7])
@pytest.mark.parametrize(
    "method", [InitMethod.normal, InitMethod.llama, InitMethod.llama_depth, InitMethod.fan_in]
)
def test_seed_and_initialization_scaling(num_bases, method):
    config = tiny_config(init_method=method)
    config.mlp_matrix_mixing.num_bases = num_bases
    model = config.build(init_device="meta")
    model.init_weights(device=torch.device("cpu"))
    initial = copy.deepcopy(model.state_dict())
    model.init_weights(device=torch.device("cpu"))
    for name, p in model.state_dict().items():
        torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
    for i, block in enumerate(model.blocks.values()):
        for proj, fan_in in [
            (block.feed_forward.w2, block.feed_forward.hidden_size),
            (block.attention.w_out, block.attention.w_out.in_features),
        ]:
            std = config.init_std
            if method == InitMethod.llama:
                std /= (2 * config.n_layers) ** 0.5
            elif method == InitMethod.llama_depth:
                std /= (2 * (i + 1)) ** 0.5
            elif method == InitMethod.fan_in:
                std = fan_in**-0.5
            torch.testing.assert_close(
                proj.coefficients.norm(), torch.tensor(std / config.init_std)
            )


def test_bias_and_device_move():
    linear = nn.Linear(3, 4, dtype=torch.float64)
    bank = MatrixBasisBank(linear, 2, 0.02)
    bank.init_weights()
    projection = MixedLinear(linear, bank)
    projection.init_weights(std=0.02)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    projection(x).sum().backward()
    torch.testing.assert_close(projection.bias.grad, torch.full_like(projection.bias, 2))
    torch.testing.assert_close(x.grad, projection.weight.detach().sum(0).expand_as(x))
    torch.testing.assert_close(projection(x), F.linear(x, projection.weight, projection.bias))
    model = tiny_config().build()
    model.init_weights(device=torch.device("cpu"))
    clone = copy.deepcopy(model).to(dtype=torch.float64)
    assert clone.blocks["0"].feed_forward.w1.bank is clone.matrix_bases["feed_forward_w1"]
    assert clone.blocks["0"].feed_forward.w1.bank is not model.matrix_bases["feed_forward_w1"]
    assert clone.blocks["0"].feed_forward.w1.weight.dtype == torch.float64


def test_optimizer_checkpoint_and_activation_recomputation(tmp_path):
    config = tiny_config()
    model = config.build()
    model.init_weights(device=torch.device("cpu"))
    reference = copy.deepcopy(model)
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ref_optim = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    before = copy.deepcopy(model.state_dict())
    for current, optimizer in [(model, optim), (reference, ref_optim)]:
        current(inputs).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name])
        if "bases" in name or "coefficients" in name:
            assert not torch.equal(value, before[name])
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optim": optim.state_dict()}, path)
    restored = config.build(init_device="meta")
    restored.to_empty(device="cpu")
    restored_optim = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    checkpoint = torch.load(path, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    restored_optim.load_state_dict(checkpoint["optim"])
    for current, optimizer in [(model, optim), (restored, restored_optim)]:
        current(inputs).square().mean().backward()
        optimizer.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


@pytest.mark.parametrize("num_bases", [0, -1, 1.5, True])
def test_invalid_num_bases(num_bases):
    with pytest.raises(OLMoConfigurationError, match="positive integer"):
        MatrixMixingConfig(num_bases)


def test_unsupported_combinations():
    config = tiny_config()
    config.block.sequence_mixer.name = AttentionType.fused
    with pytest.raises(OLMoConfigurationError, match="Q/K/V/O"):
        config.build(init_device="meta")
    config = tiny_config()
    override = copy.deepcopy(config.block)
    override.feed_forward.hidden_size *= 2
    config.block_overrides = {1: override}
    with pytest.raises(OLMoConfigurationError, match="identical shapes"):
        config.build(init_device="meta")
    model = tiny_config().build()
    for method in (model.apply_tp, model.apply_pp):
        with pytest.raises(OLMoConfigurationError, match="does not yet support"):
            method(None)


def run_distributed_mixing(mode, wrapping, checkpoint_dir):
    device = get_default_device()
    config = tiny_config()
    reference = config.build(init_device="meta")
    reference.init_weights(device=device)
    model = config.build(init_device="meta")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    if mode.endswith("_compile"):
        model.apply_compile()
    bf16 = "bf16" in mode
    if bf16:
        reference.to(dtype=torch.bfloat16)
    if mode.startswith("fsdp"):
        model.apply_fsdp(
            mesh,
            wrapping_strategy=wrapping,
            prefetch_factor=1,
            param_dtype=torch.bfloat16 if bf16 else torch.float32,
        )
    else:
        model.apply_ddp(mesh)
    model.init_weights(device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ref_optim = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    inputs = (torch.arange(16, device=device).view(2, 8) + dist.get_rank()) % config.vocab_size
    for _ in range(2):
        actual, expected = model(inputs), reference(inputs)
        torch.testing.assert_close(
            actual, expected, atol=5e-3 if bf16 else 2e-5, rtol=5e-2 if bf16 else 2e-5
        )
        actual.square().mean().backward()
        expected.square().mean().backward()
        ref_params = dict(reference.named_parameters())
        for name, p in model.named_parameters():
            grad = ref_params[name.replace("_checkpoint_wrapped_module.", "")].grad
            dist.all_reduce(grad)
            grad.div_(dist.get_world_size())
            torch.testing.assert_close(
                get_full_tensor(p.grad).float(),
                grad.float(),
                atol=5e-3 if bf16 else 2e-5,
                rtol=5e-2 if bf16 else 2e-4,
            )
        optim.step()
        ref_optim.step()
        optim.zero_grad()
        ref_optim.zero_grad()
    save_model_and_optim_state(checkpoint_dir, model, optim)
    expected_state = {n: get_full_tensor(p).detach().clone() for n, p in model.named_parameters()}
    model.init_weights(device=device)
    load_model_and_optim_state(checkpoint_dir, model, optim)
    for name, p in model.named_parameters():
        torch.testing.assert_close(get_full_tensor(p), expected_state[name])


@requires_multi_gpu
@pytest.mark.parametrize(
    "mode,wrapping",
    [
        ("fsdp", TransformerDataParallelWrappingStrategy.full),
        ("fsdp", TransformerDataParallelWrappingStrategy.fine_grained),
        ("ddp", TransformerDataParallelWrappingStrategy.full),
        ("fsdp_bf16", TransformerDataParallelWrappingStrategy.full),
        ("fsdp_bf16_compile", TransformerDataParallelWrappingStrategy.full),
    ],
)
def test_distributed_mixing(mode, wrapping, tmp_path):
    run_distributed_test(
        run_distributed_mixing,
        backend="nccl",
        start_method="spawn",
        func_args=(mode, wrapping, tmp_path / "checkpoint"),
    )


@pytest.mark.parametrize("builder", ["olmo3_1B", "olmo3_7B", "olmo3_32B"])
def test_large_model_meta_counts(builder):
    config = getattr(TransformerConfig, builder)(
        vocab_size=100352,
        attn_backend=AttentionBackendName.torch,
        mlp_matrix_mixing=MatrixMixingConfig(4),
        attn_matrix_mixing=MatrixMixingConfig(8),
    )
    assert config.num_params == config.build(init_device="meta").num_params


def test_config_cli_overrides():
    config = tiny_config(False, False).merge(
        ["mlp_matrix_mixing={num_bases: 2}", "attn_matrix_mixing={num_bases: 3}"]
    )
    assert config.mlp_matrix_mixing.num_bases == 2
    assert config.attn_matrix_mixing.num_bases == 3
    config = config.merge(["mlp_matrix_mixing.num_bases=5"])
    assert config.mlp_matrix_mixing.num_bases == 5
    with pytest.raises(OLMoConfigurationError):
        config.merge(["attn_matrix_mixing.num_bases=0"]).build(init_device="meta")


@requires_gpu
@pytest.mark.parametrize("compile_model", [False, True])
def test_gpu_amp_and_compile(compile_model):
    model = tiny_config().build(init_device="meta")
    model.init_weights(device=torch.device("cuda"), max_seq_len=8)
    reference = copy.deepcopy(model)
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    if compile_model:
        model.apply_compile()
    inputs = torch.arange(16, device="cuda").view(2, 8)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ref_optim = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual, expected = model(inputs), reference(inputs)
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-2)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        ref_params = dict(reference.named_parameters())
        for name, p in model.named_parameters():
            grad = ref_params[name.replace("_checkpoint_wrapped_module.", "")].grad
            torch.testing.assert_close(p.grad, grad, atol=5e-3, rtol=5e-2)
            assert torch.isfinite(p.grad).all()
        optim.step()
        ref_optim.step()
        optim.zero_grad()
        ref_optim.zero_grad()
