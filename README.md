# 🌩️ Scalable Ray Data Embedding Pipeline for **Qwen3-Embedding-4B**

This repository provides a **fully scalable**, **GPU-accelerated**, **Ray Data–based** embedding pipeline for generating embeddings using **Qwen3-Embedding-4B** (or any Qwen3 embedding model).

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
  - `embedding` (float32 list)
- Compatible with **Qwen/Qwen3-Embedding-0.6B**, **1.5B**, **4B**, etc.

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

---

## 🔧 Environment Variables

Caches stored in:

```
/data/<user>/HF_HOME/
```

Modify:

```python
scratch = "/data/gbhatt2/HF_HOME/"
os.environ["HF_HOME"] = scratch
```

---

## ❗ Troubleshooting

### **Model not found**
Pre-download:

```bash
huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir ./model
```

Or run in offline mode:

```bash
export TRANSFORMERS_OFFLINE=1
```

---

## 🙌 Acknowledgements

- **Ray** team  
- **Alibaba Qwen** team  
- **HuggingFace** team  
