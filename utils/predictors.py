from .meta import label_to_int
from .collators import XlmRobertaCollator
from copy import deepcopy
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
import torch
from datasets import Dataset
from tqdm.auto import trange


def predict_sentences(sentences: list[str], model: XLMRobertaForSequenceClassification, tokenizer: XLMRobertaTokenizerFast):
    collator = XlmRobertaCollator(tokenizer, label_to_int)
    data = [
        {
            "sentence": sentence,
            "label": "sport",
        }
        for sentence in sentences
    ]
    inp = collator(data)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        output = model(**inp)
    preds = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
    def find_ans(pred):
        for k, v in label_to_int.items():
            if v == pred:
                return k
    preds = [find_ans(pred) for pred in preds]
    return preds


def predict_text(text: str, model: XLMRobertaForSequenceClassification, tokenizer: XLMRobertaTokenizerFast, return_probs=True, batch_size=4):
    sentences = text.split('.')
    preds = []
    for i in range(0, len(sentences), batch_size):
        preds.extend(predict_sentences(sentences[i:i+4], model, tokenizer))
    probs = deepcopy(label_to_int)
    for k in probs.keys():
        probs[k] = preds.count(k) / len(preds)
    if return_probs:
        return probs
    else:
        ans = list(probs.items())
        ans.sort(key=lambda x: x[1])
        return ans[-1][0]


def predict_class_accuracy(dataset: Dataset, model, tokenizer):
    ans = dict()
    for key in label_to_int:
        ans[key] = {
            "total": 0,
            "pos": 0,
        }
    for i in trange(len(dataset)):
        answer = predict_text(dataset[i]["text"], model, tokenizer, return_probs=False)
        label = dataset[i]["main_topic"]
        ans[label]["total"] += 1
        if answer == label:
            ans[label]["pos"] += 1
    res = dict()
    for key in label_to_int.keys():
        res[key] = ans[key]["pos"] / ans[key]["total"]
    return res