import os
import sys
import json
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.global_config import GlobalConfig
from core.data_manager import StegoDataManager




def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ──────────────────── PPL (GPT-2) ────────────────────

_ppl_model = None
_ppl_tokenizer = None


def _load_ppl_model(model_name, device):
    global _ppl_model, _ppl_tokenizer
    if _ppl_model is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        print(f"  [PPL] Loading model: {model_name} ...")
        _ppl_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        _ppl_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        _ppl_model.eval()
        _ppl_tokenizer.pad_token = _ppl_tokenizer.eos_token
    return _ppl_model, _ppl_tokenizer


def compute_ppl(sentences, model_name="gpt2", device="cuda"):
    model, tokenizer = _load_ppl_model(model_name, device)
    ppls = []
    for sent in tqdm(sentences, desc="  PPL", leave=False):
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc.input_ids.to(device)
        if input_ids.size(1) < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
            ppl = math.exp(min(out.loss.item(), 100))
            ppls.append(ppl)
    avg = float(np.mean(ppls)) if ppls else 0.0
    return ppls, avg




_embed_models = {}  # name -> SentenceTransformer


def _load_embed_model(model_name, device):
    global _embed_models
    if model_name not in _embed_models:
        from sentence_transformers import SentenceTransformer
        print(f"  [Embed] Loading model: {model_name} ...")
        _embed_models[model_name] = SentenceTransformer(model_name, device=device)
    return _embed_models[model_name]


def get_embeddings(sentences, model_name, device, batch_size=128):
    model = _load_embed_model(model_name, device)
    embs = model.encode(sentences, batch_size=batch_size,
                        show_progress_bar=False, convert_to_numpy=True)
    return embs  # (N, d)




_bert_model = None
_bert_tokenizer = None


def _load_bert_model(model_name, device):
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        from transformers import BertModel, BertTokenizerFast
        print(f"  [KLD2] Loading BERT model: {model_name} ...")
        _bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
        _bert_model = BertModel.from_pretrained(model_name).to(device)
        _bert_model.eval()
    return _bert_model, _bert_tokenizer


def get_bert_embeddings(sentences, model_name="bert-base-uncased",
                        device="cuda", batch_size=64):
    """用 BERT [CLS] token 作为句向量。"""
    model, tokenizer = _load_bert_model(model_name, device)
    all_embs = []
    for i in tqdm(range(0, len(sentences), batch_size),
                  desc="  KLD2-BERT", leave=False):
        batch = sentences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc)
            cls_embs = out.last_hidden_state[:, 0, :]  # [CLS]
            all_embs.append(cls_embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)  # (N, 768)


# ──────────────── KLD2 ────────────────
#
#   KLD2(X||Y) = Σ_k [ log(σ_{y,k} / σ_{x,k})
#                     + (σ²_{x,k} + (μ_{x,k} - μ_{y,k})²) / (2 σ²_{y,k})
#                     - 1/2 ]

def compute_kld2(cover_embs: np.ndarray, stego_embs: np.ndarray) -> float:
    eps = 1e-8

    mu_x = np.mean(cover_embs, axis=0)
    mu_y = np.mean(stego_embs, axis=0)
    sigma_x = np.std(cover_embs, axis=0) + eps
    sigma_y = np.std(stego_embs, axis=0) + eps

    kld2 = np.sum(
        np.log(sigma_y / sigma_x)
        + (sigma_x ** 2 + (mu_x - mu_y) ** 2) / (2.0 * sigma_y ** 2)
        - 0.5
    )
    return float(kld2)


# ──────────────── SS────────────────

def compute_ss(cover_embs: np.ndarray, stego_embs: np.ndarray):
    c_norm = cover_embs / (np.linalg.norm(cover_embs, axis=1, keepdims=True) + 1e-10)
    s_norm = stego_embs / (np.linalg.norm(stego_embs, axis=1, keepdims=True) + 1e-10)
    sims = np.sum(c_norm * s_norm, axis=1)
    return sims.tolist(), float(np.mean(sims))


# ──────────────── EC ────────────────

def compute_ec(bits_lines, stego_lines):
    total_bits = sum(len(b) for b in bits_lines)
    total_words = sum(len(s.split()) for s in stego_lines)
    bpw = total_bits / total_words if total_words > 0 else 0.0
    return float(bpw), total_bits, total_words




def eval_method(method, output_dir, device, ppl_model, kld_model, ss_model):
    method_dir = StegoDataManager.method_dir(output_dir, method)

    cover_path = os.path.join(method_dir, "cover.txt")
    stego_path = os.path.join(method_dir, "stego.txt")
    bits_path  = os.path.join(method_dir, "stego_bits.txt")

    for p in [cover_path, stego_path, bits_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] {p} not found")
            return None

    cover_lines = load_lines(cover_path)
    stego_lines = load_lines(stego_path)
    bits_lines  = load_lines(bits_path)

    n = min(len(cover_lines), len(stego_lines), len(bits_lines))
    if n == 0:
        print(f"  [SKIP] Empty files for {method}")
        return None
    cover_lines = cover_lines[:n]
    stego_lines = stego_lines[:n]
    bits_lines  = bits_lines[:n]

    print(f"\n{'─'*55}")
    print(f"  Method: {method}   ({n} pairs)")
    print(f"{'─'*55}")


    _, avg_ppl = compute_ppl(stego_lines, model_name=ppl_model, device=device)


    print("  Computing BERT embeddings for KLD2 ...")
    cover_embs_kld = get_bert_embeddings(cover_lines, model_name=kld_model, device=device)
    stego_embs_kld = get_bert_embeddings(stego_lines, model_name=kld_model, device=device)
    kld2 = compute_kld2(cover_embs_kld, stego_embs_kld)


    print("  Computing RoBERTa embeddings for SS ...")
    cover_embs_ss = get_embeddings(cover_lines, model_name=ss_model, device=device)
    stego_embs_ss = get_embeddings(stego_lines, model_name=ss_model, device=device)
    _, avg_ss = compute_ss(cover_embs_ss, stego_embs_ss)


    ec_bpw, total_bits, total_words = compute_ec(bits_lines, stego_lines)

    result = {
        "method":      method,
        "n":           n,
        "PPL":         round(avg_ppl, 4),
        "KLD2":        round(kld2, 6),
        "SS":          round(avg_ss, 6),
        "EC(bpw)":     round(ec_bpw, 4),
        "total_bits":  total_bits,
        "total_words": total_words,
        "ppl_model":   ppl_model,
        "kld_model":   kld_model,
        "ss_model":    ss_model,
    }

    result_path = os.path.join(method_dir, "eval_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {result_path}")

    return result


# ══════════════════ main ══════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate: PPL / KLD2 / SS / EC")
    parser.add_argument("--method", type=str, required=True,
                        help="Comma-separated method names or 'all'")
    parser.add_argument("--ppl_model", type=str, default="gpt2",
                        help="PPL model (default: gpt2)")
    parser.add_argument("--kld_model", type=str, default="bert-base-uncased",
                        help="BERT model for KLD2 (default: bert-base-uncased)")
    parser.add_argument("--ss_model", type=str,
                        default="sentence-transformers/roberta-base-nli-mean-tokens",
                        help="Sentence model for SS (default: roberta-base-nli-mean-tokens)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = GlobalConfig()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.method.strip().lower() == "all":
        methods = []
        if os.path.isdir(config.OUTPUT_DIR):
            for d in sorted(os.listdir(config.OUTPUT_DIR)):
                if os.path.isfile(os.path.join(config.OUTPUT_DIR, d, "stego.txt")):
                    methods.append(d)
    else:
        methods = [m.strip() for m in args.method.split(",") if m.strip()]

    if not methods:
        print("No methods found. Run generation first.")
        return

    print(f"Methods   : {methods}")
    print(f"PPL model : {args.ppl_model}")
    print(f"KLD model : {args.kld_model}")
    print(f"SS model  : {args.ss_model}")
    print(f"Device    : {device}")

    results = []
    for method in methods:
        r = eval_method(method, config.OUTPUT_DIR, device,
                        args.ppl_model, args.kld_model, args.ss_model)
        if r:
            results.append(r)

    if not results:
        print("\nNo valid results.")
        return


    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"  PPL: {args.ppl_model} | KLD2: {args.kld_model} | SS: {args.ss_model}")
    print(f"{'═'*70}")
    header = f"{'Method':<12} {'PPL':>10} {'KLD2':>12} {'SS':>10} {'EC(bpw)':>10}"
    print(header)
    print(f"{'─'*12} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
    for r in results:
        print(f"{r['method']:<12} {r['PPL']:>10.2f} {r['KLD2']:>12.4f} "
              f"{r['SS']:>10.4f} {r['EC(bpw)']:>10.4f}")

    if len(results) > 1:
        avg_ppl = np.mean([r["PPL"] for r in results])
        avg_kld = np.mean([r["KLD2"] for r in results])
        avg_ss  = np.mean([r["SS"] for r in results])
        avg_ec  = np.mean([r["EC(bpw)"] for r in results])
        print(f"{'─'*12} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
        print(f"{'AVG':<12} {avg_ppl:>10.2f} {avg_kld:>12.4f} "
              f"{avg_ss:>10.4f} {avg_ec:>10.4f}")

    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()