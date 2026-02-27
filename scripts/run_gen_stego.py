import os
import sys
import importlib
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.global_config import GlobalConfig
from core.model_handler import ModelHandler
from core.data_manager import StegoDataManager, BitStreamReader
from core.sampler import BatchedStegoGenerator, internal_prompt, build_output_text


def load_method(method_name: str, config, tokenizer):
    mod = importlib.import_module(f"methods.{method_name}")
    if not hasattr(mod, "StegoMethod"):
        raise AttributeError(
            f"methods/{method_name}.py must define class StegoMethod(config, tokenizer)."
        )
    return mod.StegoMethod(config=config, tokenizer=tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="adg")
    args = parser.parse_args()

    config = GlobalConfig()
    config.setup_runtime()

    print(
        f"EMBED_SKIP_PROB={getattr(config,'EMBED_SKIP_PROB',None)}, "
        f"SKIP_FLATTEN_ALPHA={getattr(config,'SKIP_FLATTEN_ALPHA',None)}, "
        f"SKIP_DROP_TOP_N={getattr(config,'SKIP_DROP_TOP_N',None)}, "
    )

    handler = ModelHandler(config)

    print(f"Device:     {config.DEVICE}")
    print(f"Method:     {args.method}")
    print(f"Target:     {config.NUM_SENTENCES} sentences")
    print(f"Batch size: {config.BATCH_SIZE}\n")

    method = load_method(args.method, config, handler.tokenizer)


    ctx_path = StegoDataManager.context_file(config.OUTPUT_DIR, args.method)
    if not os.path.exists(ctx_path):
        raise FileNotFoundError(
            f"Context file not found: {ctx_path}\n"
            f"Run: python scripts/prepare_data.py --method {args.method}"
        )

    contexts = StegoDataManager.load_contexts(ctx_path)
    contexts = StegoDataManager.normalize_count(contexts, config.NUM_SENTENCES)

    bit_reader = BitStreamReader(config.MESSAGE_FILE)


    stego_sentences = []
    stego_bits_list = []

 
    pending_indices = list(range(config.NUM_SENTENCES))
    result_map = {}  # index -> {"text": str, "bits": str}

    pbar = tqdm(total=config.NUM_SENTENCES, desc=f"Stego ({args.method})")

    attempt_round = 0
    max_rounds = config.STEGO_MAX_ATTEMPTS



    batch_gen = BatchedStegoGenerator(handler, config, method, bit_reader)

    while pending_indices and attempt_round < max_rounds:
        attempt_round += 1

        new_pending = []

        for batch_start in range(0, len(pending_indices), config.BATCH_SIZE):
            batch_idx = pending_indices[batch_start:batch_start + config.BATCH_SIZE]
            batch_ctx = [contexts[i] for i in batch_idx]


            if hasattr(method, "reset_sentence"):
                method.reset_sentence()


            results = batch_gen.generate_batch(batch_ctx)

            for idx, res in zip(batch_idx, results):
                if res["text"] is not None:
                    result_map[idx] = res
                    pbar.update(1)
                    pbar.set_postfix(total_bits=bit_reader.total_embedded)
                else:
                    new_pending.append(idx)

        pending_indices = new_pending

    pbar.close()

    if pending_indices:
        raise RuntimeError(
            f"Failed to generate {len(pending_indices)} stego sentences after {max_rounds} rounds."
        )


    for i in range(config.NUM_SENTENCES):
        stego_sentences.append(result_map[i]["text"])
        stego_bits_list.append(result_map[i]["bits"])


    written_bits = sum(len(b) for b in stego_bits_list)
    print(f"\n{'='*40}")
    print(f"Sentences: {len(stego_sentences)}")
    print(f"Total bits (reader counter): {bit_reader.total_embedded}")
    print(f"Total bits (written): {written_bits}")
    print(f"Avg bits/sentence (written): {written_bits / len(stego_sentences):.2f}")
    print(f"{'='*40}")

    StegoDataManager.save_stego_result(args.method, stego_sentences, stego_bits_list, output_dir=config.OUTPUT_DIR)


if __name__ == "__main__":
    main()