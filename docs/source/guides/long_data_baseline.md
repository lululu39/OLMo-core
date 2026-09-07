# OLMo3-100M dense baseline on Long-Data-Collections

This experiment uses the complete `ZefanCai/Long-Data-Collections` Hugging Face
bucket as its source. Raw files are mirrored under
`/mnt/localssd/dataset/Long-Data-Collections/raw/` as lossless `.zst` streams.
The full uncompressed inventory is about 327 GB. Compression avoids requiring
that much temporary disk space. Each receipt records the remote Xet hash, raw
size, compressed size, and SHA-256 of the uncompressed bytes. Preparation stops
if the filesystem has less than 20 GiB free; it never deletes unrelated data.

```bash
uv pip install --python .venv/bin/python wandb pyarrow tokenizers zstandard
.venv/bin/python src/scripts/data/prepare_long_data.py --download-workers 32
```

The inventory is pinned on the first invocation. Completed downloads and token
shards are reused on restart; an interrupted individual download is retried from
its beginning. To inspect an original Arrow file, decompress its `.zst` stream
with Zstandard or use `zstandard.open()` and `pyarrow.ipc.open_stream()` directly.
Do not decompress the entire mirror unless sufficient disk space is available.

Preparation samples documents using SHA-256 of seed `34521` and the UTF-8 text.
The first 64 hash bits, modulo one million, assign 200 values to validation and
the next 60,000 values to training. Thus the expected document sampling rates
are 0.02% validation and 6% training. This is a sample from all Arrow shards and
all three bucket components, not a claim that the full corpus is trained on.
Repeated documents are retained within their assigned split, but identical text
cannot appear in both splits. There is no document truncation or text cleaning.
The tokenizer is `allenai/gpt-neox-olmo-dolma-v1_5`, pinned to a Hub commit and a
local SHA-256. Each document ends with EOS 50279. Output `.npy` files are raw
little-endian uint16 token streams without a NumPy header, as expected by OLMo.
The final manifest records source digests, token digests, counts and selection
settings. Dataset caches stay beside this manifest.

The model has 101,868,032 parameters (76,112,384 excluding input embeddings),
12 layers, width 512 and 8 attention heads. Both matrix mixing options are
explicitly `None`, and startup checks that no `MixedLinear` modules exist.
The 2,048-token context is shorter than OLMo3's 4,096-token local window, so all
layers use equivalent full causal SDPA attention without redundant window masks.

The default training budget is 8,192 steps × 262,144 tokens = 2,147,483,648 tokens.
The script requires enough prepared sequences to finish without repeating an
epoch. Data order seed is 34521 and initialization seed is 1337. Optimization
uses AdamW, LR 0.003, betas (0.9, 0.95), weight decay 0.1 (zero for input
embeddings), gradient clipping 1.0, z-loss multiplier 1e-5, 256 warmup steps and
cosine decay to 10% of peak LR. FSDP uses BF16 parameters and FP32 reductions;
the per-GPU microbatch is 16,384 tokens. Model compilation is enabled by default.

Set `WANDB_API_KEY` through a protected environment or credential file, and set
`WANDB_BASE_URL=https://api.wandb.ai`. Never put a key in a command, config, or Git.

```bash
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  src/scripts/train/OLMo3/long_data_baseline.py \
  --manifest /mnt/localssd/dataset/Long-Data-Collections/tokens-seed34521-train60000-val200/manifest.json \
  --save-folder /data/yibo/OLMo-core/runs/olmo3-102m-dense-longdata-2p147bt-s2048-seed1337 \
  --run-name olmo3-102m-dense-longdata-2p147bt-s2048-seed1337 \
  --entity yibozhong657-none
```

W&B logs to `weight-mixing-llm` in the specified entity. The full experiment
configuration, manifest digest and parameter counts are recorded there and in
checkpoints. Held-out language-model loss/perplexity is evaluated at startup,
every 500 steps and on finish, with a fixed 16-batch evaluation budget.
Temporary checkpoints are saved every 250 steps, permanent checkpoints every
1,000 steps and at training completion, retaining at most three permanent
checkpoints. The same command resumes model, optimizer and trainer/data-loader
state from the save folder. For subsequent mixing comparisons, reuse this exact
manifest, tokenizer, seeds, token budget, batch sizes and evaluation schedule.

Infrastructure validation uses a separately labelled synthetic fixture and
`--offline`; it is not part of the baseline or its W&B history. Once a real run
has completed startup and shows finite losses and stable throughput, estimate
completion from remaining steps and observed step time, and check near that ETA
instead of continuously polling the job.
