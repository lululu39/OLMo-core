# Cross-layer matrix mixing for pretraining

This experiment compares dense OLMo/OLMo3 pretraining with static soft parameter
sharing. For each selected projection kind, layer `l` uses

```math
W_l = \sum_{k=1}^{K} a_{l,k} B_k.
```

The bases and coefficients are randomly initialized and jointly trained. There
is no token dependence, routing, coefficient softmax, or frozen pretrained weight.
Coefficients can become positive or negative and change freely during training.

## Configuration

The two options are independent and default to `None` (dense):

```python
from olmo_core.nn.transformer import MatrixMixingConfig, TransformerConfig

config = TransformerConfig.olmo3_7B(
    vocab_size=100352,
    mlp_matrix_mixing=MatrixMixingConfig(num_bases=4),
    attn_matrix_mixing=MatrixMixingConfig(num_bases=4),
)
model = config.build(init_device="meta")
# The existing TransformerTrainModule handles wrapping and init_weights as usual.
```

Use just `mlp_matrix_mixing` for MLP mixing, just `attn_matrix_mixing` for
attention mixing, both for the combined experiment, or neither for dense.
The options also work with the other dense OLMo2/OLMo3 size builders. MLP and
attention may use different K values. They serialize through the usual model
config JSON/YAML, so checkpoint configs preserve the parameterization.

```yaml
# Fields inside the model configuration:
mlp_matrix_mixing:
  num_bases: 4
attn_matrix_mixing: null
```

For scripts using `ExperimentConfig`, enable a previously `None` option with a
whole config value, for example `--model.mlp_matrix_mixing='{num_bases: 4}'`.
After enabling it, `--model.mlp_matrix_mixing.num_bases=8` changes K. A nested
field override alone cannot construct an optional config that is still `None`.

## Code structure and ownership

- `nn/transformer/config.py`: model size builders, independent mixing options,
  validation, and actual trainable parameter counts.
- `nn/transformer/model.py`: creates the shared banks once at the model root,
  replaces selected projections as blocks are built, and initializes/shards them.
- `nn/matrix_mixing.py`: the basis bank and reusable `MixedLinear` projection.
- `nn/feed_forward.py`: `w2(activation(w1(x)) * w3(x))`, so `w1` is gate,
  `w3` is up, and `w2` is down.
- `nn/attention/__init__.py`: separate `w_q`, `w_k`, `w_v`, and `w_out` projections;
  GQA/MQA use their actual, potentially smaller, K/V output dimensions.
- `nn/transformer/init.py`: existing per-projection initialization scales are
  forwarded to coefficient initialization.

Each projection kind has a distinct bank; for example, gate and up do not share
with each other even though they have the same shape. Every layer shares the
same bank for a given kind, including sliding-window and global attention layers
when their projection shapes agree. Checkpoint names look like:

```text
matrix_bases.feed_forward_w1.bases       # (K, hidden_size, d_model)
blocks.0.feed_forward.w1.coefficients    # (K,)
blocks.1.feed_forward.w1.coefficients    # (K,)
matrix_bases.attention_w_k.bases         # (K, n_kv_heads * head_dim, d_model)
blocks.0.attention.w_k.coefficients      # (K,)
```

Layers have a non-registered reference to the bank module, not a cached parameter
reference. Consequently, banks appear once in `parameters()` and `state_dict()`,
remain connected after `to_empty()`, `.to()`, deepcopy, and checkpoint loading,
and receive the sum of gradient contributions from all layers. Optional biases
remain ordinary independent per-layer parameters. Norms, embeddings, LM head,
and optional attention gating projections retain their existing parameterization.

## Initialization and optimization

Bases use OLMo's truncated normal initialization with `init_std`. Each coefficient
vector is sampled from a normal distribution, then normalized **only at
initialization** to norm `projection_std / init_std`. Conditional on these
coefficients, each effective weight entry therefore has the same variance as the
corresponding dense initialization, up to the common truncated-normal variance
factor. Its distribution need not itself be truncated normal. This also preserves
`llama`, `llama_depth`, and `fan_in` per-projection scaling. With K=1, initial
coefficient directions are random signs; training is still unconstrained.

The bank parameters and coefficients participate in the existing optimizer and
checkpoint flow. Optimizer parameter-name overrides may need adjustment: mixed
weights are no longer named `blocks.*.*.weight`. In particular, consider a separate
coefficient learning-rate/weight-decay ablation; the implementation deliberately
uses the optimizer's configured defaults. Global basis gradients aggregate across
layers, so dense learning-rate settings are a starting point to test, not a claim
of optimization equivalence.

## Efficiency and supported training paths

For a projection with `P = out_features * in_features`, L dense layers store
`L * P` weights; mixing stores `K * P + L * K`. Bias counts are unchanged. K can
exceed L, but parameter savings require `K * P + L * K < L * P`.

Forward uses one contraction `a @ bases.flatten(1)` and one standard `F.linear`.
It does not create a `(K, tokens, hidden)` activation tensor or broadcast a
`(K, out, in)` weighted product. The contraction costs approximately `2*K*P`
FLOPs per projection invocation, while the linear costs approximately `2*T*P`
for T local tokens. There is also additional backward work. Reduced parameter
count does not imply reduced dense projection FLOPs or improved throughput.
OLMo's idealized FLOP estimate retains the effective dense projection cost and
**excludes** matrix synthesis and recomputation; use measured throughput for
compute-matched comparisons.

Ordinary autograd can retain each effective matrix for backward. Use the existing
full-block activation checkpointing option to trade recomputation for lower
activation/effective-weight memory. There is no persistent effective-weight cache
that can become stale after optimizer updates or retain graphs across steps.

Supported integration includes single-device training, AMP, DDP, and FSDP2,
including full/fine-grained block wrapping, activation checkpointing, and the
normal model/optimizer checkpoint APIs. FSDP2 keeps shared bases in the root group
and retains their unsharded values through backward; they must not be independently
wrapped per layer. This saves sharded parameter/optimizer storage but keeps a full
basis bank resident during the step. Block coefficients retain the normal block
wrapping. Compilation uses the ordinary PyTorch matmul/linear graph.

The first implementation rejects TP, PP, FP8 conversion, normalized transformers,
and mixed attention with fused QKV or recurrent sequence mixers. MLP mixing
requires standard dense MLPs. Selected projections must have identical shapes and
dtypes across layers; heterogeneous overrides fail explicitly instead of silently
creating separate sharing groups. Attention mixing can still use existing
attention compute backends with separate Q/K/V/O projections. Dense and mixed
checkpoints have different parameter schemas and cannot be loaded interchangeably.

## Suggested first experiment

Run dense, MLP-only, attention-only, and combined mixing with the same OLMo3 size,
token budget, data order, context length, optimizer, and evaluation schedule.
Sweep K (for example 1, 2, 4, 8) and multiple initialization seeds. Report actual
trainable parameters, validation loss versus tokens and wall time, tokens/second,
peak GPU memory, and basis/coefficient gradient norms. Follow with coefficient
optimizer ablations and parameter/compute-matched dense baselines. The research
question is whether this parameterization improves pretraining; this implementation
does not establish that empirical result.
