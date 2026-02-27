
# 📦 White-box Steganography

**White-box Steganography** is a modular research framework for embedding sensitive data into the text generation process of Large Language Models (LLMs). This repository provides a standardized environment for implementing and benchmarking **linguistic steganography** algorithms, emphasizing the preservation of the model’s original statistical distribution.

By decoupling **model orchestration**, **bitstream management**, and **embedding logic**, this framework enables high-throughput batch generation and rigorous evaluation across diverse linguistic metrics.

---

## 🗂️ Project Structure

```text
.
├── config/             # Global configurations
├── core/               # Model handling & sampling logic
├── data/               # Bitstream & Corpora
├── methods/            # Core steganography algorithms
└── scripts/            # Entry scripts for generation & evaluation

```

> **Note**: Place your corpora under `data/`, e.g., `data/IMDB2020.txt`.

---

## 🛠️ Key Features

* ✅ **Six Diverse Methods**: AC, ADG, Discop, iMEC, Meteor, and SparSamp.
* ⚡ **High-Efficiency Batching**: Optimized GPU inference via batched processing.
* 🔁 **Robust Synchronization**: Snapshot/restore logic for bitstream reader.
* 📊 **Built-in Metrics**: Integrated evaluation for PPL, SS, and EC/BPW.

---

## 📚 Methodology Overview

| Method | Description |
| :--- | :--- |
| **AC** | Maps secret bits into model distribution intervals via recursive arithmetic coding. |
| **Meteor** | Reduces finite-precision sampling bias to better preserve the target distribution. |
| **ADG** | Embeds data by adaptively grouping tokens based on their conditional probabilities. |
| **iMEC** | Optimizes capacity and security by framing steganography as a minimum-entropy coupling task. |
| **Discop** | Mitigates grouping-induced statistical bias using multiple distribution copies for practical security. |
| **SparSamp** | Employs sparse sampling with `O(1)` complexity for efficient, plug-and-play steganography. |

---

## 🚀 Usage Pipeline

### 1. Environment Setup

```bash
pip install torch transformers numpy tqdm sentence_transformers

```

### 2. Data Initialization

Generate the message bitstream and method-specific context prompts:

```bash
python scripts/prepare_data.py --method ac,adg,discop,imec,meteor,sparsamp --source ./data/IMDB2020.txt

```

### 3. Generation & Evaluation

Follow these steps to generate and evaluate stego-text:

* **Generate Cover (Baseline)**: `python scripts/run_gen_cover.py --method adg`
* **Generate Stego**: `python scripts/run_gen_stego.py --method adg`
* **Run Evaluation**: `python scripts/run_eval.py --method all`

---

## ⚙️ Configuration

* **Global Settings**: Managed in `config/global_config.py` (e.g., `MODEL_PATH`, `BATCH_SIZE`, `NUM_SENTENCES`).
* **Method Tuning**: Use environment variables for runtime adjustment.

**Example**:

```bash
export AC_PRECISION=24
python scripts/run_gen_stego.py --method ac


