import re
import torch

_BAD_RE = re.compile(
    r"(\bAnswer\s*:|答案\s*[:：]|____|正确|错误|（|）|"
    r"\bA\.\s+.*\bB\.\s+.*\bC\.\s+.*(\bD\.\s+.*)?)",
    re.IGNORECASE,
)


def is_bad_text(text: str) -> bool:
    return _BAD_RE.search(text) is not None


def internal_prompt(config, context: str) -> str:
    if getattr(config, "USE_INTERNAL_PROMPT", True):
        return f"{config.INTERNAL_PROMPT_PREFIX}{context}{config.INTERNAL_PROMPT_SUFFIX}"
    return f"{context}".rstrip()


def build_output_text(context: str, continuation: str) -> str:
    return (context.rstrip() + " " + continuation.lstrip()).strip()


def precompute_end_punct_ids(tokenizer, punct=".!?"):
    end_ids = set()
    vocab_size = getattr(tokenizer, "vocab_size", None) or 0
    for tid in range(vocab_size):
        s = tokenizer.decode([tid], skip_special_tokens=True)
        if s and s.rstrip().endswith(tuple(punct)):
            end_ids.add(tid)
    return end_ids


class BasicSampler:


    def __init__(self, model_handler, config):
        self.handler = model_handler
        self.config = config
        self.tokenizer = model_handler.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_punct = getattr(config, "STOP_PUNCT", ".!?")
        self._end_punct_ids = None

    def _compute_end_punct_ids(self):
        self._end_punct_ids = precompute_end_punct_ids(self.tokenizer, self.stop_punct)

    @torch.no_grad()
    def generate_one_sentence(self, context: str, temperature_override: float = None) -> str:

        prompt = internal_prompt(self.config, context)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.DEVICE)
        logits, past = self.handler.get_logits_and_past(input_ids, past_key_values=None, use_cache=True)

        generated_ids = []
        cont_text = ""

        if self._end_punct_ids is None:
            self._compute_end_punct_ids()


        orig_temp = self.config.TEMPERATURE
        if temperature_override is not None:
            self.config.TEMPERATURE = temperature_override

        try:
            for step in range(self.config.MAX_NEW_TOKENS):
                step_logits = logits
                if step < self.config.MIN_NEW_TOKENS and self.eos_token_id is not None:
                    step_logits = step_logits.clone()
                    step_logits[:, self.eos_token_id] = -float("inf")

                probs, indices = self.handler.process_probs(step_logits)
                next_pos = torch.multinomial(probs, num_samples=1)
                next_id = indices[next_pos].item()

                generated_ids.append(next_id)
                piece = self.tokenizer.decode([next_id], skip_special_tokens=True)
                cont_text += piece

                if step + 1 >= self.config.MIN_NEW_TOKENS:
                    if next_id in self._end_punct_ids:
                        break
                if self.eos_token_id is not None and next_id == self.eos_token_id and (step + 1) >= self.config.MIN_NEW_TOKENS:
                    break

                curr = torch.tensor([[next_id]], device=self.config.DEVICE)
                logits, past = self.handler.get_logits_and_past(curr, past_key_values=past, use_cache=True)
        finally:
            self.config.TEMPERATURE = orig_temp

        out = build_output_text(context, cont_text)
        out = out.replace("\n", " ").replace("\r", " ")
        return out


class BatchedCoverGenerator:


    def __init__(self, model_handler, config):
        self.handler = model_handler
        self.config = config
        self.tokenizer = model_handler.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.end_punct_ids = precompute_end_punct_ids(self.tokenizer, config.STOP_PUNCT)

    @torch.no_grad()
    def generate_batch(self, contexts: list) -> list:

        batch_size = len(contexts)
        if batch_size == 0:
            return []

        device = self.config.DEVICE
        pad_id = self.tokenizer.pad_token_id or 0

        prompts = [internal_prompt(self.config, ctx) for ctx in contexts]
        input_ids, attention_mask = self.handler.batch_encode(prompts)

        logits, past = self.handler.batch_get_logits_and_past(
            input_ids, attention_mask
        )

        generated_ids = [[] for _ in range(batch_size)]
        done = [False] * batch_size

        for step in range(self.config.MAX_NEW_TOKENS):
            if all(done):
                break

            step_logits = logits.clone()

            if step < self.config.MIN_NEW_TOKENS and self.eos_token_id is not None:
                step_logits[:, self.eos_token_id] = -float("inf")

            next_token_ids = []
            for i in range(batch_size):
                if done[i]:
                    next_token_ids.append(pad_id)
                    continue

                probs, indices = self.handler.process_probs(step_logits[i:i + 1])
                pos = torch.multinomial(probs, num_samples=1)
                nid = indices[pos].item()
                generated_ids[i].append(nid)
                next_token_ids.append(nid)

                if step + 1 >= self.config.MIN_NEW_TOKENS:
                    if nid in self.end_punct_ids:
                        done[i] = True
                    if self.eos_token_id is not None and nid == self.eos_token_id:
                        done[i] = True

            if all(done):
                break

            next_input = torch.tensor(next_token_ids, device=device).unsqueeze(1)
            new_mask_col = torch.tensor(
                [0 if done[i] else 1 for i in range(batch_size)],
                device=device, dtype=attention_mask.dtype
            ).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

            logits, past = self.handler.batch_get_logits_and_past(
                next_input, attention_mask, past_key_values=past
            )

        results = []
        for i in range(batch_size):
            if not generated_ids[i]:
                results.append((contexts[i], None))
                continue

            cont_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
            full_text = build_output_text(contexts[i], cont_text)
            full_text = full_text.replace("\n", " ").replace("\r", " ")

            if is_bad_text(full_text):
                results.append((contexts[i], None))
            else:
                results.append((contexts[i], full_text))

        return results


class BatchedStegoGenerator:


    def __init__(self, model_handler, config, method, bit_reader):
        self.handler = model_handler
        self.config = config
        self.method = method
        self.bit_reader = bit_reader
        self.tokenizer = model_handler.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.end_punct_ids = precompute_end_punct_ids(self.tokenizer, config.STOP_PUNCT)

    @torch.no_grad()
    def generate_batch(self, contexts: list):

        import random as _random

        batch_size = len(contexts)
        if batch_size == 0:
            return []

        device = self.config.DEVICE
        pad_id = self.tokenizer.pad_token_id or 0
        skip_prob = float(getattr(self.config, "EMBED_SKIP_PROB", 0.0) or 0.0)

        snapshots = [self.bit_reader.snapshot() for _ in range(batch_size)]
        generated_ids = [[] for _ in range(batch_size)]
        bits_parts = [[] for _ in range(batch_size)]
        done = [False] * batch_size

        prompts = [internal_prompt(self.config, ctx) for ctx in contexts]
        input_ids, attention_mask = self.handler.batch_encode(prompts)

        logits, past = self.handler.batch_get_logits_and_past(
            input_ids, attention_mask
        )

        for step in range(self.config.MAX_NEW_TOKENS):
            if all(done):
                break

            step_logits = logits.clone()

            if step < self.config.MIN_NEW_TOKENS and self.eos_token_id is not None:
                step_logits[:, self.eos_token_id] = -float("inf")

            next_token_ids = []
            for i in range(batch_size):
                if done[i]:
                    next_token_ids.append(pad_id)
                    continue

                probs, indices = self.handler.process_probs(step_logits[i:i + 1])

                do_skip = (_random.random() < skip_prob)
                if do_skip:
                    token_id = _sample_skip(probs, indices, self.config)
                    bits = ""
                else:
                    token_id, bits = self.method.step(probs, indices, self.bit_reader)

                generated_ids[i].append(token_id)
                if bits:
                    bits_parts[i].append(bits)
                next_token_ids.append(token_id)

                if step + 1 >= self.config.MIN_NEW_TOKENS:
                    if token_id in self.end_punct_ids:
                        done[i] = True
                    if self.eos_token_id is not None and token_id == self.eos_token_id:
                        done[i] = True

            if all(done):
                break

            next_input = torch.tensor(next_token_ids, device=device).unsqueeze(1)
            new_mask_col = torch.tensor(
                [0 if done[i] else 1 for i in range(batch_size)],
                device=device, dtype=attention_mask.dtype
            ).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

            logits, past = self.handler.batch_get_logits_and_past(
                next_input, attention_mask, past_key_values=past
            )

        results = []
        for i in range(batch_size):
            n_gen = len(generated_ids[i])
            sentence_bits = "".join(bits_parts[i])

            if self.config.MIN_NEW_TOKENS <= n_gen <= self.config.MAX_NEW_TOKENS and n_gen > 0:
                cont_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
                full_text = build_output_text(contexts[i], cont_text)
                full_text = full_text.replace("\n", " ").replace("\r", " ")
                results.append({
                    "context": contexts[i],
                    "text": full_text,
                    "bits": sentence_bits,
                })
            else:
                self.bit_reader.restore(snapshots[i])
                results.append({
                    "context": contexts[i],
                    "text": None,
                    "bits": "",
                })

        return results


def _sample_skip(sorted_probs, sorted_indices, config):
    probs = sorted_probs

    drop_n = int(getattr(config, "SKIP_DROP_TOP_N", 0) or 0)
    if drop_n > 0 and probs.numel() > drop_n:
        probs = probs.clone()
        probs[:drop_n] = 0.0
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = torch.ones_like(probs) / probs.numel()

    alpha = float(getattr(config, "SKIP_FLATTEN_ALPHA", 1.0) or 1.0)
    if alpha != 1.0:
        probs = torch.pow(probs, alpha)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = torch.ones_like(probs) / probs.numel()

    pos = torch.multinomial(probs, num_samples=1).item()
    return int(sorted_indices[pos].item())