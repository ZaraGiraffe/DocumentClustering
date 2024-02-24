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

    def load_sentences(self, min_len=-1, max_len=-1):
        def gen():
            for i in range(len(self.dataset)):
                example = self.dataset[i]
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

        self.dataset = Dataset.from_generator(gen)
        return self.dataset

    def shuffle(self, seed=42):
        self.dataset = self.dataset.shuffle(seed=seed)
        return self.dataset