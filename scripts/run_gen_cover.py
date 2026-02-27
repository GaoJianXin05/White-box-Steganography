import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.global_config import GlobalConfig
from core.model_handler import ModelHandler
from core.data_manager import StegoDataManager
from core.sampler import BatchedCoverGenerator, BasicSampler, is_bad_text


def main():
    parser = argparse.ArgumentParser(description="Generate cover text for a specific method.")
    parser.add_argument("--method", type=str, required=True, help="Method name, e.g. ac")
    args = parser.parse_args()

    config = GlobalConfig()
    config.setup_runtime()

    handler = ModelHandler(config)

    ctx_path = StegoDataManager.context_file(config.OUTPUT_DIR, args.method)
    if not os.path.exists(ctx_path):
        raise FileNotFoundError(
            f"Context file not found: {ctx_path}\n"
            f"Run: python scripts/prepare_data.py --method {args.method}"
        )

    contexts = StegoDataManager.load_contexts(ctx_path)
    contexts = StegoDataManager.normalize_count(contexts, config.NUM_SENTENCES)

    print(f"Method:     {args.method}")
    print(f"Contexts:   {ctx_path}  ({len(contexts)} lines)")
    print(f"Target:     {config.NUM_SENTENCES} cover sentences")
    print(f"Batch size: {config.BATCH_SIZE}\n")

    batch_gen = BatchedCoverGenerator(handler, config)
    single_gen = BasicSampler(handler, config)

    result_map = {}  # index -> text


    pending = list(range(config.NUM_SENTENCES))

    pbar = tqdm(total=config.NUM_SENTENCES, desc=f"Cover ({args.method})")

    batch_rounds = 0
    max_batch_rounds = 3 

    while pending and batch_rounds < max_batch_rounds:
        batch_rounds += 1
        next_pending = []

        for batch_start in range(0, len(pending), config.BATCH_SIZE):
            batch_idx = pending[batch_start:batch_start + config.BATCH_SIZE]
            batch_ctx = [contexts[i] for i in batch_idx]

            results = batch_gen.generate_batch(batch_ctx)

            for idx, (ctx, text) in zip(batch_idx, results):
                if text is not None:
                    result_map[idx] = text
                    pbar.update(1)
                else:
                    next_pending.append(idx)

        pending = next_pending


    if pending:
        print(f"\n  [Stage 2] Retrying {len(pending)} failed sentences with single generation (higher temperature)...")

        retry_temperatures = [
            config.TEMPERATURE,       
            config.TEMPERATURE * 1.3, 
            config.TEMPERATURE * 1.6, 
            2.0,                      
        ]

        still_pending = []
        for idx in tqdm(pending, desc="Single retry"):
            ok = False
            for temp in retry_temperatures:
                for _ in range(config.COVER_MAX_ATTEMPTS):
                    text = single_gen.generate_one_sentence(contexts[idx], temperature_override=temp)
                    if not is_bad_text(text):
                        result_map[idx] = text
                        pbar.update(1)
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                still_pending.append(idx)

        pending = still_pending


    if pending:
        print(f"\n  [Stage 3] Force-accepting {len(pending)} stubborn sentences (bad_text filter bypassed)...")
        for idx in pending:
            text = single_gen.generate_one_sentence(contexts[idx], temperature_override=2.0)

            result_map[idx] = text
            pbar.update(1)
            print(f"    [WARN] Forced context #{idx}: {text[:80]}...")

    pbar.close()


    cover_texts = [result_map[i] for i in range(config.NUM_SENTENCES)]

    if len(cover_texts) != config.NUM_SENTENCES:
        raise RuntimeError("Cover line count mismatch.")

    StegoDataManager.save_cover_result(cover_texts, output_dir=config.OUTPUT_DIR, method=args.method)
    print(f"\nDone. Saved {len(cover_texts)} cover sentences.")


if __name__ == "__main__":
    main()