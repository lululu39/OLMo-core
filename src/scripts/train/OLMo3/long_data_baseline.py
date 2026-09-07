"""Train the dense OLMo3-100M baseline on a prepared Long-Data-Collections manifest.

Launch with torchrun. Credentials are read only from WANDB_API_KEY; never put them
in command line overrides. The manifest, model, optimizer and data-order settings
are recorded in each checkpoint and in the public W&B run.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.matrix_mixing import MixedLinear
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all


def build_configs(opts):
    """Construct baseline configs and validate the data/tokenizer contract."""
    data = json.loads(opts.manifest.read_text())
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    settings = data["settings"]
    if (
        settings["tokenizer"],
        settings["vocab_size"],
        settings["eos_token_id"],
        settings["dtype"],
    ) != (tokenizer.identifier, tokenizer.vocab_size, tokenizer.eos_token_id, "uint16"):
        raise ValueError("Manifest tokenizer does not match this baseline")
    paths = {}
    for split in ("train", "validation"):
        paths[split] = []
        for entry in data["files"]:
            path = Path(entry["paths"][split])
            if path.stat().st_size != entry[split + "_tokens"] * 2:
                raise ValueError(f"Token file size does not match manifest: {path}")
            if entry[split + "_tokens"] >= (opts.sequence_length if split == "train" else 1):
                paths[split].append(str(path))
        if not paths[split]:
            raise ValueError(f"No usable {split} sequences")
    available = sum(Path(p).stat().st_size // (2 * opts.sequence_length) for p in paths["train"])
    available_tokens = available * opts.sequence_length
    if available_tokens < opts.steps * opts.global_batch_size:
        raise ValueError("Prepared data is too small for the requested single-pass token budget")
    work = opts.manifest.parent / "dataset-cache" / f"seq{opts.sequence_length}"
    # At context <= 4096 the OLMo3 local windows are equivalent to full attention.
    # Omitting the redundant mask lets SDPA use its fused causal attention kernel.
    if opts.sequence_length > 4096:
        raise ValueError("This baseline uses context <= 4096")
    model = TransformerConfig.olmo3_100M(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.torch,
        sliding_window=None,
        mlp_matrix_mixing=None,
        attn_matrix_mixing=None,
    )
    dataset = NumpyFSLDatasetConfig(
        paths=paths["train"],
        tokenizer=tokenizer,
        sequence_length=opts.sequence_length,
        work_dir=str(work),
    )
    loader = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=34521,
        num_workers=4,
    )
    module = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.microbatch_tokens,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=3e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=CosWithWarmup(warmup_steps=min(256, max(1, opts.steps // 10)), alpha_f=0.1),
        max_grad_norm=1.0,
        z_loss_multiplier=1e-5,
        compile_model=not opts.no_compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )
    trainer = (
        TrainerConfig(
            save_folder=str(opts.save_folder),
            work_dir=str(opts.save_folder / "work"),
            max_duration=Duration.steps(opts.steps),
            metrics_collect_interval=10,
            cancel_check_interval=100,
            save_overwrite=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                max_checkpoints=3,
                pre_train_checkpoint=False,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                entity=opts.entity,
                project="weight-mixing-llm",
                group="olmo3-100m-longdata-seed1337",
                tags=["dense", "baseline", "olmo3-100m", "long-data-collections", "seed1337"],
                enabled=not opts.offline,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=paths["validation"],
                    metadata=[{"label": "longdata-heldout"}] * len(paths["validation"]),
                    tokenizer=tokenizer,
                    sequence_length=opts.sequence_length,
                    work_dir=str(work),
                ),
                eval_interval=500,
                eval_duration=Duration.steps(16),
                eval_on_startup=True,
                eval_on_finish=True,
            ),
        )
    )
    config = dict(
        model=model.as_config_dict(),
        dataset=dataset.as_config_dict(),
        data_loader=loader.as_config_dict(),
        train_module=module.as_config_dict(),
        trainer=trainer.as_config_dict(),
        init_seed=1337,
        num_params=model.num_params,
        num_non_embedding_params=model.num_non_embedding_params,
        manifest=str(opts.manifest),
        manifest_sha256=hashlib.sha256(opts.manifest.read_bytes()).hexdigest(),
        data_settings=settings,
        train_token_budget=opts.steps * opts.global_batch_size,
        available_train_tokens=available_tokens,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        training_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    return model, dataset, loader, module, trainer, config


def main():
    """Build, resume if applicable, and train the baseline with native OLMo Trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save-folder", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--entity", default="yibozhong657-none")
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=262144)
    parser.add_argument("--microbatch-tokens", type=int, default=16384)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Disable W&B for a local smoke test")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()
    model_cfg, dataset_cfg, loader_cfg, module_cfg, trainer_cfg, config = build_configs(opts)
    if opts.dry_run:
        print(json.dumps(config, indent=2))
        return
    prepare_training_environment()
    try:
        seed_all(1337)
        model = model_cfg.build(init_device="meta")
        if any(isinstance(m, MixedLinear) for m in model.modules()):
            raise RuntimeError("Dense baseline unexpectedly contains mixed projections")
        actual_params = sum(p.numel() for p in model.parameters())
        if actual_params != config["num_params"]:
            raise ValueError("Parameter count mismatch")
        module = module_cfg.build(model)
        dataset = dataset_cfg.build()
        loader = loader_cfg.build(dataset, dp_process_group=module.dp_process_group)
        trainer = trainer_cfg.build(module, loader)
        trainer.callbacks["config_saver"].config = config
        trainer.callbacks["wandb"].config = config
        if get_rank() == 0:
            opts.save_folder.mkdir(parents=True, exist_ok=True)
            (opts.save_folder / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
            print(f"Dense baseline verified: {actual_params:,} parameters", flush=True)
        trainer.maybe_load_checkpoint()
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
