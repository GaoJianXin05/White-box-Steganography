import os
import random
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class GlobalConfig:
    """
    Global configuration shared by cover/stego generation and evaluation.
    """

    # ---- Model / IO ----
    MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B")
    CONTEXT_FILE = os.getenv("CONTEXT_FILE", "data/contexts.txt")
    MESSAGE_FILE = os.getenv("MESSAGE_FILE", "data/message_bits.txt")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

    # ---- Experiment size ----
    NUM_SENTENCES = int(os.getenv("NUM_SENTENCES", "11000"))
    EXPECTED_MESSAGE_BITS = int(os.getenv("EXPECTED_MESSAGE_BITS", "1000000"))

    # ---- Batch generation ----
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

    # ---- Decoding length constraints (new tokens / continuation only) ----
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
    MIN_NEW_TOKENS = int(os.getenv("MIN_NEW_TOKENS", "10"))

    # ---- Sampling parameters (algorithm-level) ----
    TEMPERATURE = float(os.getenv("TEMPERATURE", "1.1"))
    TOP_K = int(os.getenv("TOP_K", "0"))
    TOP_P = float(os.getenv("TOP_P", "0.99"))

    # ---- Performance candidate cap (implementation-level) ----
    CANDIDATE_TOP_K = int(os.getenv("CANDIDATE_TOP_K", "4096"))
    FORCE_CANDIDATE_CAP = False

    # ---- Rate control (optional shared knobs) ----
    EMBED_SKIP_PROB = float(os.getenv("EMBED_SKIP_PROB", "0"))
    SKIP_FLATTEN_ALPHA = float(os.getenv("SKIP_FLATTEN_ALPHA", "1"))
    SKIP_DROP_TOP_N = int(os.getenv("SKIP_DROP_TOP_N", "0"))

    # ---- Device / dtype ----
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    PROB_DTYPE = torch.float32

    # ---- Retry / filtering ----
    COVER_MAX_ATTEMPTS = int(os.getenv("COVER_MAX_ATTEMPTS", "30"))
    STEGO_MAX_ATTEMPTS = int(os.getenv("STEGO_MAX_ATTEMPTS", "30"))

    # ---- Prompting (must be consistent for cover & stego) ----
    USE_INTERNAL_PROMPT = False
    INTERNAL_PROMPT_PREFIX = (
        "Write a English sentence continuing with.\nPrompt: "
    )
    INTERNAL_PROMPT_SUFFIX = "\nSentence:"

    # ---- Stop rules (optional) ----
    STOP_ON_EOS = False
    STOP_PUNCT = ".?!"

    # ---- Reproducibility ----
    SEED = int(os.getenv("SEED", "-1"))
    DETERMINISTIC = bool(int(os.getenv("DETERMINISTIC", "0")))

    @classmethod
    def validate(cls):
        if cls.NUM_SENTENCES <= 0:
            raise ValueError("NUM_SENTENCES must be > 0")
        if cls.EXPECTED_MESSAGE_BITS <= 0:
            raise ValueError("EXPECTED_MESSAGE_BITS must be > 0")
        if cls.MIN_NEW_TOKENS < 0 or cls.MAX_NEW_TOKENS <= 0:
            raise ValueError("MIN_NEW_TOKENS/MAX_NEW_TOKENS invalid")
        if cls.MIN_NEW_TOKENS > cls.MAX_NEW_TOKENS:
            raise ValueError("MIN_NEW_TOKENS > MAX_NEW_TOKENS")
        if cls.TEMPERATURE <= 0:
            raise ValueError("TEMPERATURE must be > 0")
        if cls.TOP_K < 0:
            raise ValueError("TOP_K must be >= 0")
        if not (0.0 < cls.TOP_P <= 1.0):
            raise ValueError("TOP_P must be in (0,1]")
        if cls.CANDIDATE_TOP_K <= 0:
            raise ValueError("CANDIDATE_TOP_K must be > 0")
        if cls.CANDIDATE_TOP_K < 128:
            raise ValueError("CANDIDATE_TOP_K too small (<128), will hurt capacity")
        if not (0.0 <= cls.EMBED_SKIP_PROB < 1.0):
            raise ValueError("EMBED_SKIP_PROB must be in [0,1)")
        if cls.SKIP_FLATTEN_ALPHA <= 0:
            raise ValueError("SKIP_FLATTEN_ALPHA must be > 0")
        if cls.SKIP_DROP_TOP_N < 0:
            raise ValueError("SKIP_DROP_TOP_N must be >= 0")
        if cls.USE_INTERNAL_PROMPT and (not cls.INTERNAL_PROMPT_PREFIX or not cls.INTERNAL_PROMPT_SUFFIX):
            raise ValueError("INTERNAL_PROMPT_PREFIX/SUFFIX required when USE_INTERNAL_PROMPT=True")
        if cls.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be > 0")

    @staticmethod
    def _random_seed_32bit() -> int:
        return int.from_bytes(os.urandom(4), "little", signed=False)

    @classmethod
    def setup_runtime(cls):
        cls.validate()

        if int(cls.SEED) < 0:
            cls.SEED = cls._random_seed_32bit()

        random.seed(int(cls.SEED))
        np.random.seed(int(cls.SEED))
        torch.manual_seed(int(cls.SEED))
        if cls.DEVICE == "cuda":
            torch.cuda.manual_seed_all(int(cls.SEED))

        if cls.DETERMINISTIC:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

        torch.set_float32_matmul_precision("high")
        print(f"[Seed] SEED={cls.SEED}  (set SEED env to reproduce; SEED=-1 for random each run)")