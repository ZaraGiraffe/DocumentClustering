import numpy as np
import pandas as pd
from tqdm.auto import trange
from tqdm.auto import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data import Lemmatizer
from datasets import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
import json
import argparse


description = """
The program creates the most frequent relevant words from IMDB dataset
"""


"""
python tools/create_termins.py --num 50 --file termins.json --lang en --stopwords ./stopwords/en.json --dataset ./doc.parquet
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TerminsCreator", description=description)
    parser.add_argument("--num", help="the number of words to create", default=300, type=int)
    parser.add_argument("--file", help="the json file path where to put the termins", default="./termins.json")
    parser.add_argument("--lang", help="the language of the dataset lemmatizer", default="en")
    parser.add_argument("--stopwords", help="the path to file with stopwords", default="none")
    parser.add_argument("--dataset", help="the hf dataset in parquet format", default="none")

    args = parser.parse_args()

    dataset = Dataset.from_parquet(args.dataset)

    lemmatizer = Lemmatizer(args.lang)

    vectorizer = TfidfVectorizer()

    texts = []
    for text in tqdm(dataset["text"]):
        texts.append(lemmatizer.lemmatize_text(text))

    X = vectorizer.fit_transform(texts)

    ans = []
    for i in trange(X.shape[1]):
        ans.append(X.getcol(i).count_nonzero())
    word_nonzero = np.array(ans)

    if args.stopwords != "none":
        stopwords = json.load(open(args.stopwords, "r", encoding="utf-8"))
        stop_mask = np.array([word in stopwords for word in vectorizer.get_feature_names_out()])
    else:
        stop_mask = np.full(len(vectorizer.get_feature_names_out()), False)

    df = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "count": word_nonzero * np.invert(stop_mask)
    })

    df = df.sort_values("count", ascending=False)

    termins = df["word"][:args.num].to_list()
    json.dump(termins, open(args.file, "w", encoding="utf-8"))