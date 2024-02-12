import numpy as np
import pandas as pd
from tqdm.notebook import trange
from tqdm.notebook import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data import Lemmatizer
from utils.loaders import IMDBDataset

from sklearn.feature_extraction.text import TfidfVectorizer
import json
import nltk
from nltk.corpus import stopwords
import argparse


description = """
The program creates the most frequent relevant words from IMDB dataset
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TerminsCreator", description=description)
    parser.add_argument("--num", help="the number of words to create", default=300, type=int)
    parser.add_argument("--file", help="the json file path where to put the termins", default="./termins.json")
    args = parser.parse_args()

    imdb_dataset = IMDBDataset.load()

    lemmatizer = Lemmatizer("en")

    vectorizer = TfidfVectorizer()

    texts = []
    for text in tqdm(imdb_dataset["train"]["text"]):
        texts.append(lemmatizer.lemmatize_text(text))

    X = vectorizer.fit_transform(texts)

    ans = []
    for i in trange(X.shape[1]):
        ans.append(X.getcol(i).count_nonzero())

    X = vectorizer.fit_transform(texts)

    ans = []
    for i in trange(X.shape[1]):
        ans.append(X.getcol(i).count_nonzero())
    word_nonzero = np.array(ans)

    nltk.download('stopwords')

    stop_mask = np.array([word in stopwords.words("english") for word in vectorizer.get_feature_names_out()])

    df = pd.DataFrame({
        "word": vectorizer.get_feature_names_out(),
        "count": word_nonzero * np.invert(stop_mask)
    })

    termins = df["word"][:args.num].to_list()
    json.dump(termins, open(args.file, "w"))