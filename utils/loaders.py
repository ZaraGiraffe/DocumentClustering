from datasets import load_dataset, Dataset
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


class DocDataset:
    def __init__(self, dataset_dir, lang):
        self.dataset_dir = dataset_dir
        self.lang = lang
        self.dataset = self.load()

    def load(self) -> Dataset:
        """
        load my own DocDataset to hf dataset
        :return: Dataset
        """
        def gen():
            for yaml_file in os.listdir(os.path.join(self.dataset_dir, "annotations")):
                with open(os.path.join(self.dataset_dir, "annotations", yaml_file), "r") as file:
                    anot = yaml.safe_load(file)
                if anot["lang"] != self.lang:
                    continue
                with open(os.path.join(self.dataset_dir, "docs", anot["filename"]), "r") as file:
                    anot["text"] = file.read()
                yield anot
        dataset = Dataset.from_generator(gen)
        return dataset

class DatasetUA:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset = self.load()

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
