"""
Abstractive Module: Fine-tuned AraT5 on XL-Sum.
Step 1: Instruction-based prompt construction (anti-hallucination).
"""
import os
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

from .utils import find_model_dir

# ⚠️ Keep "summarize: " — it must match your training prefix.
# The Arabic instruction constrains generation to source facts.
PROMPT_PREFIX = "summarize: "
INSTRUCTION = "لخص النص التالي مستخدماً فقط المعلومات الواردة فيه دون إضافة حقائق جديدة. النص:\n"


class AraT5Summarizer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_path = find_model_dir(model_path)
        print(f"Loading AraT5 from: {model_path}")

        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"config.json not found in {model_path}")

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(model_path, "tokenizer.json"),
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>"
        )       
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()

        self.model.generation_config.no_repeat_ngram_size = 2
        self.model.generation_config.early_stopping = True
        self.model.generation_config.max_length = 128
        self.model.generation_config.num_beams = 4

    def build_prompt(self, text: str, use_instruction: bool = True) -> str:
        """Step 1: structured prompt — instruction + text."""
        if use_instruction:
            return PROMPT_PREFIX + INSTRUCTION + text
        return PROMPT_PREFIX + text

    def _generate(self, prompt: str, max_length: int, num_beams: int) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text: str, max_length: int = 128,
                  num_beams: int = 4, use_instruction: bool = False) -> str:
        prompt = self.build_prompt(text, use_instruction=use_instruction)
        return self._generate(prompt, max_length, num_beams)