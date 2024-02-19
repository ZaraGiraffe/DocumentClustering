import torch
from transformers import BertModel, BertTokenizer
from .data import Lemmatizer

def get_embeddings_from_sentence(
        sentence: str,
        model: BertModel,
        tokenizer: BertTokenizer,
        lemmatizer: Lemmatizer,
        termins: list[str],
        cuda: bool = False,
):
    """
    :param sentence: given sentence to process
    :param model: bert model
    :param tokenizer: bert tokenizer
    :param lemmatizer: lemmatizer, that lemmatizes the text
    :param termins: list of termins
    :param cuda: if true puts the model to cuda and makes calculations on cuda
    :return:
    """
##########################################################################
    if cuda:
        if torch.cuda.is_available():
            model.to("cuda")
##########################################################################
    bert_embeddings = []
    zero_pair = torch.tensor([0, 0])
    bases_dict = lemmatizer.get_word_bases(sentence)
    with torch.no_grad():
        for base_dict in bases_dict:
            token_dict = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt")
            tokens_pos = []
            if base_dict["base"] in termins:
                left = base_dict["offset"]
                right = left + len(base_dict["word"]) - 1
                for i, pos in enumerate(token_dict["offset_mapping"][0]):
                    if (pos == zero_pair).all() or (pos < left).all() or (pos > right).all():
                        continue
                    tokens_pos.append(i)
            if tokens_pos:
                token_dict.pop("offset_mapping")
##########################################################################
                if cuda:
                    if torch.cuda.is_available():
                        token_dict = {k: v.cuda() for k, v in token_dict.items()}
##########################################################################
                model_output = model(**token_dict).last_hidden_state[0]
                for pos in tokens_pos:
                    bert_embeddings.append(model_output[pos].cpu())
    return bert_embeddings


def get_embeddings_from_document(
        document: str,
        model: BertModel,
        tokenizer: BertTokenizer,
        lemmatizer: Lemmatizer,
        termins: list[str],
        cuda: bool = False,
):
    """
    :param sentence: given document to process
    :param model: bert model
    :param tokenizer: bert tokenizer
    :param lemmatizer: lemmatizer, that lemmatizes the text
    :param termins: list of termins
    :param cuda: if true puts the model to cuda and makes calculations on cuda
    :return:
    """
    sentences = document.split('.')
    embeddings = []
    for sentence in sentences:
        embeddings.extend(get_embeddings_from_sentence(sentence, model, tokenizer, lemmatizer, termins, cuda))
    return embeddings