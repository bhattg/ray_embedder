# 🌩️ Scalable Ray Data Embedding Pipeline**

This repository provides a **fully scalable**, **GPU-accelerated**, **Ray Data–based** embedding pipeline for generating embeddings.

The pipeline is designed for **large-scale text embedding generation** on multi-CPU and multi-GPU nodes.

---

## 🚀 Features

- **End-to-end distributed pipeline** built on Ray Data  
- **Tokenizer stage (CPU)**: parallelized over all CPU cores  
- **Embedding stage (GPU)**: parallelized across all visible GPUs  
- Supports ⚡ **streaming execution**, **zero-copy Arrow batches**, and **parquet output**    
- Saves output parquet containing:
  - `prompt_id`
  - `text`
  - `embedding`

---

## 🧩 Architecture Overview

```
┌─────────────────────────────┐
│  JSONL / Parquet Inputs     │
└──────────────┬──────────────┘
               │ ray.data.read_json()
               ▼
┌─────────────────────────────┐
│   TokenizationStage (CPU)   │
│  • HuggingFace fast tokenizer│
│  • Left padding to max_length│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   EmbeddingStage (GPU)      │
│  • HF AutoModel             │
│  • fp16/fp32                │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   Save parquet (streaming)   │
└─────────────────────────────┘
```

---

## 📦 Installation

```bash
pip install ray[default]
pip install transformers
pip install huggingface_hub
pip install torch
pip install pandas pyarrow
```

Make sure CUDA + PyTorch with GPU support is installed.

---

## 📁 Input Format

Input must be one or many `prompt_text.jsonl` files containing:

```json
{"prompt_id": "id123", "text": "hello world"}
{"prompt_id": "id124", "text": "some text"}
```

The script automatically discovers all such files recursively.

---

## ▶️ Usage

### **Basic Run**

```bash
python embed.py   --output_dir ./embeddings/qwen3-4b/   --model_name Qwen/Qwen3-Embedding-4B   --batch_size 8   --dtype fp16
```

### **What the script will do**

- Recursively locate all `prompt_text.jsonl` files  
- Initialize Ray  
- Download/cache Qwen3 model (HuggingFace snapshot)  
- Run CPU tokenization at massive scale  
- Run GPU embedding across all GPUs  
- Save split parquet shards to `output_dir`

---

## 🛠️ Pipeline Tuning

### CPU/GPU parallelism

```python
num_cpus = os.cpu_count()
num_gpus = torch.cuda.device_count()
```

Tokenization concurrency:

```python
concurrency=(1, 2 * num_cpus)
```

Embedding concurrency:

```python
concurrency=(1, num_gpus)
num_gpus=1
```

Tune depending on system.

---

## 🧪 Output Parquet Schema

```
prompt_id: string
text: string
embedding: list<float>
```

---

## 📊 Performance Notes

- Fast tokenizer batches (size 512) → **very high CPU throughput**  
- Embeddings computed in **fp16** → **2–4× GPU speedup**  
- Streaming parquet writing → avoids memory blowup  
- Ray Data executor keeps memory bounded for large corpora
- If one has multiple compute nodes for GPU, change the concurrency argument according to [this](https://docs.ray.io/en/latest/train/user-guides/using-gpus.html).
