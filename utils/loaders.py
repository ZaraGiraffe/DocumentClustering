from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
import torch
import huggingface_hub as hub
import yaml
import os

class IMDBDataset:
    def __init__(self) -> None:
        self.dataset = IMDBDataset.load()

    @staticmethod
    def load() -> Dataset:
        """
        downloads dataset from here 
        https://huggingface.co/datasets/imdb
        :return: Dataset
        """
        dataset = load_dataset("imdb")
        return dataset

class DatasetUA:
    def __init__(self, dataset_dir=None, hf_repo=None):
        """
        :param dataset_dir:
        :param from_hf:
        """
        if dataset_dir is None and hf_repo is None:
            raise Exception("dataset_dir is None and hf_repo is None")
        if dataset_dir:
            self.dataset_dir = dataset_dir
            self.dataset = self.load()
        else:
            self.dataset = load_dataset(hf_repo)
            self.dataset = concatenate_datasets([self.dataset["train"], self.dataset["test"]])



    def load(self) -> Dataset:
        """
        load my own DocDataset to hf dataset
        :return: Dataset
        """
        def gen():
            for yaml_file in os.listdir(os.path.join(self.dataset_dir)):
                if ".yaml" not in yaml_file:
                    continue
                with open(os.path.join(self.dataset_dir, yaml_file), "r") as file:
                    anot = yaml.safe_load(file)
                with open(os.path.join(self.dataset_dir, anot["filename"]), "r") as file:
                    anot["text"] = file.read()
                yield anot
        dataset = Dataset.from_generator(gen)
        return dataset


def convert_to_sequences(dataset: Dataset, min_len=-1, max_len=-1):
    """
    Converts each text to sentences
    :param min_len: minimum required length of the sentence to be included
    :param max_len: maximum required length of the sentence to be included
    :return:
    """

    def gen():
        for i in range(len(dataset)):
            example = dataset[i]
            label = example["filename"].split("_")[0]
            sentences = example["text"].split('.')
            for sen in sentences:
                if min_len != -1 and len(sen) < min_len:
                    continue
                if max_len != -1 and len(sen) > max_len:
                    continue
                yield dict(
                    sentence=sen,
                    label=label
                )

    dataset = Dataset.from_generator(gen)
    return dataset


def load_pretrained(checkpoint_name: str):
    config = XLMRobertaConfig.from_pretrained("FacebookAI/xlm-roberta-base")
    config.num_labels = 15
    model_pr = XLMRobertaForSequenceClassification(config)

    api = hub.HfApi()
    api.hf_hub_download(
        repo_id="Zarakun/ukrainian_news_classification",
        repo_type="model",
        filename="checkpoints/{}".format(checkpoint_name),
        local_dir="./"
    )

    model_pr.load_state_dict(torch.load("./checkpoints/{}".format(checkpoint_name), map_location="cpu"))
    return model_pr