from datasets import load_dataset


class IMDBDataset:
    def __init__(self) -> None:
        self.dataset = IMDBDataset.load()

    @staticmethod
    def load():
        """
        downloads dataset from here 
        https://huggingface.co/datasets/imdb
        """
        dataset = load_dataset("imdb")
        return dataset
