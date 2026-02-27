import os
import sys
import argparse
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.global_config import GlobalConfig


def generate_context_file(output_path, num_lines=2000, source_file=None):
    contexts = []

    if source_file and os.path.exists(source_file):
        with open(source_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if len(line.split()) > 5 and line.strip()]
        sampled = random.choices(lines, k=num_lines)
        for line in sampled:
            words = line.split()
            prompt_len = random.randint(1, 5)
            contexts.append(" ".join(words[:prompt_len]))
    else:
        starters = [
            "The", "In the", "A", "There is", "It was", "On", "Before", "After",
            "She", "He", "They", "We", "I", "However,", "Although", "Because",
            "According to", "Despite", "During", "Once upon a time",
        ]
        nouns = [
            "weather", "government", "movie", "game", "system", "life", "world",
            "man", "woman", "child", "company", "study", "report", "day", "night",
            "history", "story", "problem", "solution", "idea",
        ]
        verbs = [
            "is", "was", "has", "had", "can", "could", "will", "would",
            "seems", "looks", "became", "remained", "started", "ended",
        ]

        for _ in range(num_lines):
            s = random.choice(starters)
            if s in ["She", "He", "They", "We", "I"]:
                contexts.append(f"{s} {random.choice(verbs)}")
            elif s in ["The", "A"]:
                contexts.append(f"{s} {random.choice(nouns)} {random.choice(verbs)}")
            else:
                contexts.append(s)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in contexts[:num_lines]:
            f.write(c.replace("\n", " ").replace("\r", " ") + "\n")
    print(f"  Contexts saved: {output_path}  ({num_lines} lines)")


def generate_bit_stream(output_path, length=100000):
    bits = np.random.randint(0, 2, size=length, dtype=np.int8)
    bit_str = "".join(bits.astype(str).tolist())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bit_str)
    print(f"  Bit stream saved: {output_path}  ({length} bits)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare per-method contexts and global bit stream.")
    parser.add_argument(
        "--method", type=str, required=True,
        help="Comma-separated method names, e.g. ac,adg"
    )
    parser.add_argument(
        "--source", type=str, default="./data/IMDB2020.txt",
        help="Source corpus for context prompts"
    )
    args = parser.parse_args()

    cfg = GlobalConfig()
    cfg.setup_runtime()

    methods = [m.strip() for m in args.method.split(",") if m.strip()]


    print("\n[1] Generating global bit stream ...")
    generate_bit_stream(cfg.MESSAGE_FILE, length=cfg.EXPECTED_MESSAGE_BITS)


    print(f"\n[2] Generating per-method contexts ({cfg.NUM_SENTENCES} lines each) ...")
    for method in methods:
        ctx_path = os.path.join(cfg.OUTPUT_DIR, method, "contexts.txt")
        print(f"\n  --- {method} ---")
        generate_context_file(ctx_path, num_lines=cfg.NUM_SENTENCES, source_file=args.source)

    print(f"\n[done] Methods: {methods}")
    print(f"  Output dir: {cfg.OUTPUT_DIR}")