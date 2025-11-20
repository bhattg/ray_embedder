

#!/usr/bin/env python3
"""
Fully scalable Ray Data embedding pipeline for Qwen3-Embedding-4B.

- Uses `pandas`/`pyarrow` batches for speed
- Tokenization parallelized over all CPU cores
- Embedding parallelized over all visible GPUs
- Uses last_token_pool + L2 normalization (Qwen3-Embedding-4B spec)
- Saves parquet with id, combined, embedding
"""

import os
from ray.data import DataContext

scratch = "/data/gbhatt2/HF_HOME/"
os.environ["HF_HOME"] = scratch
# os.environ["TRANSFORMERS_CACHE"] = f"{scratch}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{scratch}/datasets"
os.environ["HF_METRICS_CACHE"] = f"{scratch}/metrics"
os.environ["HF_MODULES_CACHE"] = f"{scratch}/modules"
import time
import numpy as np
import pyarrow as pa
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
import ray


# ============================================================
# Qwen3 pooling + normalization utilities
# ============================================================
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lengths]


# ============================================================
# Tokenization Stage (CPU)
# ============================================================
class TokenizerStage:
    def __init__(self, model_name: str, max_length: int = 8192):
        # start = time.time()
        # print(f"[{time.strftime('%X')}] Initializing tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", use_fast=True, local_files_only=True
        )
        self.max_length = max_length
        # print(f"[{time.strftime('%X')}] Tokenizer ready in {time.time()-start:.1f}s")

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        texts = batch["text"].tolist(); 
        try:    
            assert all(isinstance(t, str) for t in texts)
        except:
            texts =[str(t)  for t in texts]

        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        batch["input_ids"] = toks["input_ids"].tolist()
        batch["attention_mask"] = toks["attention_mask"].tolist()
        return batch


# ============================================================
# Embedding Stage (GPU)
# ============================================================
class EmbeddingStage:
    def __init__(self, model_name: str, dtype="fp16"):
        # start = time.time()
        # print(f"[{time.strftime('%X')}] Loading Qwen3 embedding model: {model_name}")
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        # print(f"[{time.strftime('%X')}] Model ready in {time.time()-start:.1f}s")

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # batch = batch.drop(columns=["messages"])  # drop text to save memory
        # convert tokenized batch (lists) -> torch tensors
        assert "prompt_id" in batch.columns, "prompt_id column missing"
        input_ids = batch["input_ids"].tolist()
        attention_mask = batch["attention_mask"].tolist()

        max_len = max(len(x) for x in input_ids)
        bs = len(input_ids)
        ids_np = np.zeros((bs, max_len), dtype=np.int64)
        mask_np = np.zeros((bs, max_len), dtype=np.int64)

        for i in range(bs):
            l = len(input_ids[i])
            ids_np[i, :l] = input_ids[i]
            mask_np[i, :l] = attention_mask[i]

        toks = {
            "input_ids": torch.tensor(ids_np, device=self.device),
            "attention_mask": torch.tensor(mask_np, device=self.device),
        }

        with torch.inference_mode():
            out = self.model(**toks)
            pooled = last_token_pool(out.last_hidden_state, toks["attention_mask"])
            # pooled = F.normalize(pooled, p=2, dim=1)
            emb = pooled.detach().cpu().numpy().astype(np.float32)

        # batch = batch[["id", "combined"]].copy()
        batch["embedding"] = emb.tolist()

        return batch[["prompt_id", "text", "embedding"]]

# ============================================================
# Main Ray Data Pipeline
# ============================================================
def main(parquet_path, output_dir, model_name="Qwen/Qwen3-Embedding-0.6B",
         batch_size=32, dtype="fp16"):
    # ctx = init_ray_auto(num_cpus=40, num_gpus=4, target_obj_store_gb=300)    
    ray.init(
        ignore_reinit_error=True,
    )
    ctx = DataContext.get_current()
    ctx.execution_options.preserve_order = False
    ctx.use_streaming_executor = True
    ctx.issue_detectors_config.high_memory_detector_config.detection_time_interval_s = -1  # disables the repetitive warning

    local_path = snapshot_download(model_name)
    print(f"Model cached at {local_path}")
    
    ds = ray.data.read_json(parquet_path,  lines=True)


    # subset_fraction=0.1
    # total_rows = ds.count()    
    # subset_size = max(1, int(total_rows * subset_fraction))
    # ds = ds.random_shuffle().limit(subset_size)

    print(f"Loaded dataset with {ds.count()} rows")

    # Auto-detect system parallelism
    num_cpus = os.cpu_count() or 8
    num_gpus = torch.cuda.device_count() or 1
    print(f"[System] Using {num_cpus} CPU cores and {num_gpus} GPUs")
 
    # CPU Tokenization (using constructor per worker)
    embedded = ds.map_batches(
        TokenizerStage,
        fn_constructor_args=(local_path,),
        batch_size=512,
        batch_format="pandas",
        concurrency=(1, 2*num_cpus),
    ).map_batches(
        EmbeddingStage,
        fn_constructor_args=(local_path, "float16"),
        batch_size=batch_size,
        batch_format="pandas",
        concurrency=(1, num_gpus),
        num_gpus=1,
    ).map_batches(
        lambda df: df.assign(text=df["text"].astype(str)),
        batch_format="pandas"
    )
    embedded = embedded.select_columns(["prompt_id", "text", "embedding"])

    # Save directly to parquet (streamed)
    embedded.write_parquet(output_dir, min_rows_per_file=10_000)
    print(f"[✓] Saved embeddings to {output_dir}")

    ray.shutdown()


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    import glob
    parser.add_argument("--output_dir", default="./embeddings/qwen3-embedding-4b/")
    parser.add_argument("--model_name", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    args = parser.parse_args()
    jsonl_files = glob.glob(os.path.join("./", "**", "prompt_text.jsonl"), recursive=True)
    main(
        parquet_path=jsonl_files,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
