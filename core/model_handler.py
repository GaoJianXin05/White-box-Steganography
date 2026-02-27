import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            torch_dtype=config.MODEL_DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(config.DEVICE)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id


        self.tokenizer.padding_side = "left"


    @torch.no_grad()
    def get_logits_and_past(self, input_ids, past_key_values=None, use_cache=True):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values


    @torch.no_grad()
    def batch_get_logits_and_past(self, input_ids, attention_mask, past_key_values=None):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits[:, -1, :], outputs.past_key_values


    def batch_encode(self, texts: list):

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        return (
            encoded["input_ids"].to(self.config.DEVICE),
            encoded["attention_mask"].to(self.config.DEVICE),
        )

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.to(torch.float32)
        if self.config.TEMPERATURE and float(self.config.TEMPERATURE) != 1.0:
            logits = logits / float(self.config.TEMPERATURE)
        return logits

    @torch.no_grad()
    def process_probs(self, logits: torch.Tensor):

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logits = self._apply_temperature(logits)
        vocab = logits.size(-1)

        cand_k = int(self.config.CANDIDATE_TOP_K) if getattr(self.config, "CANDIDATE_TOP_K", None) else vocab
        cand_k = min(max(cand_k, 1), vocab)

        if not getattr(self.config, "FORCE_CANDIDATE_CAP", True):
            if (self.config.TOP_K and self.config.TOP_K > 0) or (self.config.TOP_P is not None and self.config.TOP_P < 1.0):
                pass
            else:
                cand_k = vocab

        topk_logits, topk_indices = torch.topk(logits, k=cand_k, dim=-1)

        if self.config.TOP_K and self.config.TOP_K > 0:
            k2 = min(int(self.config.TOP_K), cand_k)
            topk_logits = topk_logits[:, :k2]
            topk_indices = topk_indices[:, :k2]

        probs = F.softmax(topk_logits, dim=-1).to(self.config.PROB_DTYPE)

        sorted_probs, order = torch.sort(probs, descending=True, dim=-1)
        sorted_indices = topk_indices.gather(-1, order)

        if self.config.TOP_P is not None and float(self.config.TOP_P) < 1.0:
            p = float(self.config.TOP_P)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = int((cum < p).sum(dim=-1).item()) + 1
            cutoff = max(1, min(cutoff, sorted_probs.size(-1)))
            sorted_probs = sorted_probs[..., :cutoff]
            sorted_indices = sorted_indices[..., :cutoff]
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        return sorted_probs.squeeze(0), sorted_indices.squeeze(0)

    @torch.no_grad()
    def batch_process_probs(self, batch_logits: torch.Tensor):

        results = []
        for i in range(batch_logits.size(0)):
            sp, si = self.process_probs(batch_logits[i:i+1])
            results.append((sp, si))
        return results