import torch


def get_embeddings_from_sentence(sentence: str, model, tokenizer, lemmatizer, termins: list[str]):
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
                model_output = model(**token_dict).last_hidden_state[0]
                for pos in tokens_pos:
                    bert_embeddings.append(model_output[pos])
    return bert_embeddings


def get_embeddings_from_document(document: str, model, tokenizer, lemmatizer, termins: list[str]):
    sentences = document.split('.')
    embeddings = []
    for sentence in sentences:
        embeddings.extend(get_embeddings_from_sentence(sentence, model, tokenizer, lemmatizer, termins))
    return embeddings