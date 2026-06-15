"""
Abstractive Module: Fine-tuned AraT5 on XL-Sum.
"""
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .utils import find_model_dir


class AraT5Summarizer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = find_model_dir(model_path)
        print(f"Loading AraT5 from: {model_path}")
        
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

        self.model.generation_config.no_repeat_ngram_size = 2
        self.model.generation_config.early_stopping = True
        self.model.generation_config.max_length = 128
        self.model.generation_config.num_beams = 4

    def summarize(self, text: str, max_length: int = 128) -> str:
        inputs = self.tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)