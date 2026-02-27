````markdown
# 📦 White-box Steganography

**White-box Steganography** is a modular research framework for embedding sensitive data into the text generation process of Large Language Models (LLMs). This repository provides a standardized environment for implementing and benchmarking **linguistic steganography** algorithms, emphasizing preservation of the model’s original statistical distribution.

By decoupling **model orchestration**, **bitstream management**, and **embedding logic**, this framework enables high-throughput batch generation and rigorous evaluation across diverse linguistic metrics.

---

## 🗂️ Project Structure

```text
.
├── methods/
│   ├── ac.py
│   ├── adg.py
│   ├── discop.py
│   ├── imec.py
│   ├── meteor.py
│   ├── sparsamp.py
│   └── __init__.py
├── core/
│   ├── data_manager.py
│   ├── model_handler.py
│   └── sampler.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_gen_cover.py
│   ├── run_gen_stego.py
│   └── run_eval.py
├── config/
│   └── global_config.py
└── data/
    ├── message_bits.txt
    └── corpora/ (e.g., IMDB2020)
````

> 📌 Put your corpora under `data/`, e.g. `data/IMDB2020.txt` or `data/IMDB2020/`.

---

## 🛠️ Key Features

* ✅ Six representative linguistic steganography methods: **AC**, **ADG**, **Discop**, **iMEC**, **Meteor**, **SparSamp**
* ⚡ Batched generation for efficient GPU inference
* 🔁 Bitstream snapshot/restore for robust sentence-level retries
* 📊 Built-in evaluation: **PPL**, **SS**, **EC/BPW**

---

## 📚 Methodology Overview

| Method       | Description                                                                                     |
| :----------- | :---------------------------------------------------------------------------------------------- |
| **AC**       | Arithmetic Coding-based interval partitioning with deterministic quantization.                  |
| **ADG**      | Adaptive Dynamic Grouping that partitions the vocabulary to balance security and capacity.      |
| **Discop**   | Distribution Copies utilizing Huffman recursion and PRNG-based rotation.                        |
| **iMEC**     | iterative Minimum-Entropy Coupling that manages per-sample belief states for optimal coupling.  |
| **Meteor**   | A masking-based approach utilizing HMAC-DRBG for cryptographically grounded security.           |
| **SparSamp** | Sparse Sampling logic designed to minimize statistical divergence during the embedding process. |

---

## 🚀 Usage Pipeline

### 1) Environment Setup

```bash
pip install torch transformers numpy tqdm sentence_transformers
```

### 2) Data Initialization

Generate the message bitstream and method-specific contexts:

```bash
python scripts/prepare_data.py --method ac,adg,discop,imec,meteor,sparsamp --source ./data/IMDB2020.txt
```

### 3) Cover Generation (Baseline)

```bash
python scripts/run_gen_cover.py --method adg
```

### 4) Stego Generation

```bash
python scripts/run_gen_stego.py --method adg
```

### 5) Evaluation

```bash
python scripts/run_eval.py --method all
```

---

## ⚙️ Configuration

Global settings live in `config/global_config.py` (e.g., `MODEL_PATH`, `BATCH_SIZE`, `NUM_SENTENCES`, sampling params).
Method-specific behavior can be tuned via environment variables.

Example:

```bash
export AC_PRECISION=24
python scripts/run_gen_stego.py --method ac
```

```
```
