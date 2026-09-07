"""Mirror a HF bucket losslessly and prepare a reproducible document-hash sample.

Raw files are stored as Zstandard streams to avoid staging the uncompressed bucket.
Training and validation membership depends only on the text and sampling seed, so
identical documents in different bucket components cannot cross the split boundary.
Install ``pyarrow tokenizers zstandard huggingface_hub requests`` before running.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import zstandard as zstd

BUCKET = "ZefanCai/Long-Data-Collections"
TOKENIZER = "allenai/gpt-neox-olmo-dolma-v1_5"
EOS = 50279
VOCAB = 50280


def write_json(path, value):
    """Atomically publish a JSON manifest or receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def mirror_file(root, item):
    """Download, hash and losslessly compress one file, with bounded retries."""
    target = Path(root) / "raw" / (item["path"] + ".zst")
    receipt = Path(str(target) + ".json")
    if target.exists() and receipt.exists():
        result = json.loads(receipt.read_text())
        if (
            result["xetHash"] == item["xetHash"]
            and target.stat().st_size == result["compressed_bytes"]
        ):
            return result
        raise ValueError(f"Existing mirror does not match inventory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(target) + ".part")
    url = f"https://huggingface.co/buckets/{BUCKET}/resolve/{quote(item['path'], safe='/')}"
    for attempt in range(5):
        try:
            digest = hashlib.sha256()
            size = 0
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError(f"Expected full response, received {response.status_code}")
                with partial.open("wb") as out:
                    with zstd.ZstdCompressor(level=3).stream_writer(out) as writer:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if shutil.disk_usage(root).free < 20 * 1024**3:
                                raise OSError("Stopping before disk fills: less than 20 GiB free")
                            digest.update(chunk)
                            size += len(chunk)
                            if size > item["size"]:
                                raise ValueError("Remote file grew after inventory was captured")
                            writer.write(chunk)
            if size != item["size"]:
                raise ValueError(f"Incomplete download: {size} != {item['size']}")
            partial.replace(target)
            result = dict(item, sha256=digest.hexdigest(), compressed_bytes=target.stat().st_size)
            write_json(receipt, result)
            return result
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def tokenize_file(task):
    """Verify a compressed Arrow shard and emit deterministic uint16 token streams."""
    import pyarrow as pa
    from tokenizers import Tokenizer

    root, item, settings = task
    root = Path(root)
    name = hashlib.sha256(item["path"].encode()).hexdigest()[:20]
    outdir = root / settings["output_dir"]
    receipt = outdir / "receipts" / f"{name}.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["settings"] != settings or result["source_sha256"] != item["sha256"]:
            raise ValueError("Tokenization settings or source changed")
        for split in ("train", "validation"):
            f = outdir / split / f"{name}.npy"
            if f.stat().st_size != result[split + "_tokens"] * 2:
                raise ValueError(f"Invalid token file: {f}")
        return result
    tokenizer = Tokenizer.from_file(str(root / "tokenizer" / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counts = {"train": 0, "validation": 0}
    docs = {"train": 0, "validation": 0}
    hashes = {split: hashlib.sha256() for split in counts}
    writers = {}
    for split in counts:
        p = outdir / split / f"{name}.npy.part"
        p.parent.mkdir(parents=True, exist_ok=True)
        writers[split] = p.open("wb")
    seed = str(settings["seed"]).encode() + b"\0"
    source = root / "raw" / (item["path"] + ".zst")
    # Check the full reconstructed file against its download digest before using it.
    digest = hashlib.sha256()
    with zstd.open(source, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError(f"Corrupt compressed mirror: {source}")
    try:
        with zstd.open(source, "rb") as stream:
            reader = pa.ipc.open_stream(stream)
            for batch in reader:
                selected = {"train": [], "validation": []}
                for text in batch.column(batch.schema.get_field_index("text")).to_pylist():
                    if not text:
                        continue
                    value = (
                        int.from_bytes(hashlib.sha256(seed + text.encode()).digest()[:8], "big")
                        % 1_000_000
                    )
                    if value < settings["validation_ppm"]:
                        selected["validation"].append(text)
                    elif value < settings["validation_ppm"] + settings["train_ppm"]:
                        selected["train"].append(text)
                for split, texts in selected.items():
                    for start in range(0, len(texts), 32):
                        encodings = tokenizer.encode_batch(
                            texts[start : start + 32], add_special_tokens=False
                        )
                        for encoding in encodings:
                            tokens = encoding.ids + [EOS]
                            if max(tokens) >= VOCAB:
                                raise ValueError("Out-of-vocabulary token")
                            data = np.asarray(tokens, dtype="<u2").tobytes()
                            writers[split].write(data)
                            hashes[split].update(data)
                            counts[split] += len(tokens)
                            docs[split] += 1
    finally:
        for writer in writers.values():
            writer.close()
    for split in counts:
        (outdir / split / f"{name}.npy.part").replace(outdir / split / f"{name}.npy")
    result = dict(
        source=item["path"],
        source_sha256=item["sha256"],
        settings=settings,
        **{split + "_tokens": count for split, count in counts.items()},
        documents=docs,
        token_sha256={split: digest.hexdigest() for split, digest in hashes.items()},
        paths={split: str(outdir / split / f"{name}.npy") for split in counts},
    )
    write_json(receipt, result)
    return result


def main():
    """Mirror the complete bucket, then prepare a reusable training manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/mnt/localssd/dataset/Long-Data-Collections")
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--tokenizer-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=34521)
    parser.add_argument("--train-ppm", type=int, default=60000)
    parser.add_argument("--validation-ppm", type=int, default=200)
    opts = parser.parse_args()
    opts.root.mkdir(parents=True, exist_ok=True)
    inventory_path = opts.root / "inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        response = requests.get(f"https://huggingface.co/api/buckets/{BUCKET}/tree", timeout=60)
        response.raise_for_status()
        inventory = response.json()
        while response.links.get("next"):
            response = requests.get(response.links["next"]["url"], timeout=60)
            response.raise_for_status()
            inventory.extend(response.json())
        inventory = sorted((x for x in inventory if x["type"] == "file"), key=lambda x: x["path"])
        write_json(inventory_path, inventory)
    print(
        f"Mirroring {len(inventory)} files, {sum(x['size'] for x in inventory) / 1e9:.2f} GB raw",
        flush=True,
    )
    start = time.monotonic()
    mirrored = []
    with ThreadPoolExecutor(max_workers=opts.download_workers) as pool:
        futures = [pool.submit(mirror_file, opts.root, item) for item in inventory]
        for future in as_completed(futures):
            mirrored.append(future.result())
            total = sum(x["size"] for x in mirrored)
            print(
                f"Downloaded {len(mirrored)}/{len(inventory)}: {total / 1e9:.2f} GB raw, {total / max(1, time.monotonic()-start) / 1e6:.1f} MB/s",
                flush=True,
            )
    mirrored.sort(key=lambda x: x["path"])
    write_json(opts.root / "mirror-manifest.json", dict(bucket=BUCKET, files=mirrored))
    from huggingface_hub import HfApi, hf_hub_download

    tokenizer_dir = opts.root / "tokenizer"
    revision_file = tokenizer_dir / "revision.json"
    if revision_file.exists():
        revision = json.loads(revision_file.read_text())["revision"]
    else:
        revision = HfApi().model_info(TOKENIZER).sha
        write_json(revision_file, dict(identifier=TOKENIZER, revision=revision))
    hf_hub_download(TOKENIZER, "tokenizer.json", revision=revision, local_dir=tokenizer_dir)
    tokenizer_sha = hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest()
    settings = dict(
        seed=opts.seed,
        train_ppm=opts.train_ppm,
        validation_ppm=opts.validation_ppm,
        tokenizer=TOKENIZER,
        tokenizer_revision=revision,
        tokenizer_sha256=tokenizer_sha,
        output_dir=f"tokens-seed{opts.seed}-train{opts.train_ppm}-val{opts.validation_ppm}",
        dtype="uint16",
        eos_token_id=EOS,
        vocab_size=VOCAB,
        selection="sha256(str(seed) + NUL + utf8(text))[:8] big endian modulo 1000000; validation then train intervals",
        duplicate_policy="retain repeated documents within a split; identical text always has the same split",
    )
    tasks = [
        (str(opts.root), item, settings) for item in mirrored if item["path"].endswith(".arrow")
    ]
    results = []
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = "4"
    with ProcessPoolExecutor(max_workers=opts.tokenizer_workers) as pool:
        futures = [pool.submit(tokenize_file, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            print(
                f"Tokenized {len(results)}/{len(tasks)} shards; train tokens={sum(r['train_tokens'] for r in results):,}",
                flush=True,
            )
    results.sort(key=lambda x: x["source"])
    manifest = dict(
        settings=settings,
        files=results,
        train_tokens=sum(x["train_tokens"] for x in results),
        validation_tokens=sum(x["validation_tokens"] for x in results),
    )
    write_json(opts.root / settings["output_dir"] / "manifest.json", manifest)
    print(f"READY: {opts.root / settings['output_dir'] / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
