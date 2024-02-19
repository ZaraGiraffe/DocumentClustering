import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data import Lemmatizer
from utils.clustering import BertCluster, BertHierarchicalClustering
from utils.bert import get_embeddings_from_document

from transformers import BertModel, BertTokenizerFast
from datasets import Dataset

import json
import random
from tqdm.auto import tqdm, trange


description = """
this program clusterizes documents
"""

"""
 python tools/make_clusters.py --dataset doc.parquet --termins ./termins.json --lang en --max_samples 20 --log_level 2 --seed 0 --clusters 2 --log_res ./log_res.txt --dataset ./doc.parquet 
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Clustering", description=description)
    parser.add_argument("--dataset", help="hf dataset in parquet format", default="none")
    parser.add_argument("--termins", help="the json file path from where to get the termins", default="./termins.json")
    parser.add_argument("--lang", help="the language of the dataset lemmatizer", default="en")
    parser.add_argument("--max_samples", help="the maximum number of samples to cluster", default=50, type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--log_res", help="where to output the results of clustering", default="log_res.txt")
    parser.add_argument("--log_level", help="how much output to give", default=1, type=int)
    parser.add_argument("--clusters", help="how much clusters to create", default=10, type=int)

    args = parser.parse_args()

    dataset = Dataset.from_parquet(args.dataset)
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    lemmatizer = Lemmatizer(args.lang)
    termins = json.load(open(args.termins))

    hierarchical_clustering = BertHierarchicalClustering(distance_type="euclidian")

    random.seed(args.seed)
    numbers = list(range(len(dataset)))
    num_samples = min(len(dataset), args.max_samples)
    numbers = sorted(random.sample(numbers, num_samples))

    with open(args.log_res, "w"):
        pass

    for i in tqdm(numbers):
        embeddings = get_embeddings_from_document(dataset[i]["text"], model, tokenizer, lemmatizer, termins)
        label = f"[{i}] " + dataset[i]["main_topic"]
        cluster = BertCluster(embeddings, i, label)
        hierarchical_clustering.add_cluster(cluster)
        if args.log_level >= 1:
            with open(args.log_res, "a") as file:
                file.write(f"[{i}], embeddings: {len(cluster.embeddings)}" + '\n')

    for i in trange(num_samples - args.clusters):
        if args.log_level >= 2:
            with open(args.log_res, "a") as file:
                file.write('-' * 10 + f" before {i} merge " + '-' * 10 + '\n')
                clusters = list(hierarchical_clustering.clusters.values())
                for cluster in clusters:
                    file.write(str(cluster.labels) + '\n')
        hierarchical_clustering.reduce_one_cluster()

    with open(args.log_res, "a") as file:
        file.write('-' * 10 + f" answer " + '-' * 10 + '\n')
        clusters = list(hierarchical_clustering.clusters.values())
        for cluster in clusters:
            file.write(str(cluster.labels) + '\n')