from dataclasses import dataclass
import torch
from transformers import XLMRobertaTokenizerFast


@dataclass
class XlmRobertaCollator:
    tokenizer: XLMRobertaTokenizerFast
    label_to_int: dict

    def __call__(self, data: list[dict]) -> dict:
        sentences = [exm["sentence"] for exm in data]
        labels = torch.tensor([self.label_to_int[exm["label"]] for exm in data])
        tokenized = self.tokenizer(sentences, padding=True, return_tensors="pt")
        tokenized["labels"] = labels
        return tokenized