import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import huggingface_hub as hub
import numpy as np
import os
from tqdm.auto import tqdm, trange


class TrainConfig:
    model_name: str = None
    tensor_board_dir: str = "./tensorboard"
    hf_repo: str = "Zarakun/ukrainian_news_classification"
    epochs: int = 2
    device: str = "cuda"
    optim_param: dict = {"lr": 0.00005, "weight_decay": 0.01}
    tune_params: str = "whole"  # "whole" - fine-tune the whole model, "head" - fine-tune only head
    tensorboard_dir: str = "./runs"
    save_repo_dir: str = "checkpoints"
    token: str = "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"


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            config: TrainConfig
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        if config.tune_params == "whole":
            tune_params = model.parameters()
        elif config.tune_params == "head":
            tune_params = model.classifier.parameters()
        else:
            raise Exception("the wrong tune_params parameter")
        self.optimizer = optimizer(tune_params, **config.optim_param)
        self.config = config
        self.start_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.model_name = "model_" + self.start_time
        self.log_dir = config.tensorboard_dir + "/" + self.model_name
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.api = hub.HfApi()

    def prepare_batch(self, batch):
        return {
            k: v.to(self.config.device)
            for k, v in batch.items()
        }

    def train_step(self, batch, batch_num: int = None):
        batch = self.prepare_batch(batch)
        output = self.model(**batch)
        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()
        self.calc_train_batch_metrics(output, batch)

    def val_step(self, batch, batch_num: int = None):
        batch = self.prepare_batch(batch)
        with torch.no_grad():
            output = self.model(**batch)
        self.calc_val_batch_metrics(output, batch)

    def calc_val_batch_metrics(self, output, batch):
        loss = output.loss.detach().cpu().numpy()
        logits = output.logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        labels = batch["labels"].cpu().numpy()
        if not hasattr(self, "val_accuracy"):
            self.val_accuracy = {
                "true": [],
                "pred": []
            }
        self.val_accuracy["pred"].extend(preds.tolist())
        self.val_accuracy["true"].extend(labels.tolist())
        if not hasattr(self, "val_losses"):
            self.val_losses = []
        self.val_losses.append(loss.tolist())

    def calc_train_batch_metrics(self, output, batch):
        loss = output.loss.detach().cpu().numpy()
        logits = output.logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)
        labels = batch["labels"].cpu().numpy()
        if not hasattr(self, "train_accuracy"):
            self.train_accuracy = {
                "true": [],
                "pred": []
            }
        self.train_accuracy["pred"].extend(preds.tolist())
        self.train_accuracy["true"].extend(labels.tolist())
        if not hasattr(self, "train_losses"):
            self.train_losses = []
        self.train_losses.append(loss.tolist())

    def aggregate_train_metrics(self):
        return {
            "train_loss": np.average(self.train_losses),
            "train_accuracy": (np.array(self.train_accuracy["pred"]) == np.array(
                self.train_accuracy["true"])).sum() / len(self.train_accuracy["pred"])
        }

    def aggregate_val_metrics(self):
        return {
            "val_loss": np.average(self.val_losses),
            "val_accuracy": (np.array(self.val_accuracy["pred"]) == np.array(self.val_accuracy["true"])).sum() / len(
                self.val_accuracy["pred"])
        }

    def refresh_train_metrics(self):
        del self.train_accuracy
        del self.train_losses

    def refresh_val_metrics(self):
        del self.val_accuracy
        del self.val_losses

    def save_train_metrics(self, epoch_num):
        metrics = self.aggregate_train_metrics()
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch_num)

    def save_val_metrics(self, epoch_num):
        metrics = self.aggregate_val_metrics()
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch_num)

    def train_epoch(self, epoch_num: int = None):
        with tqdm(self.train_loader) as tqdm_train_loader:
            for i, batch in enumerate(tqdm_train_loader):
                self.train_step(batch, i)
                tqdm_train_loader.set_description(f"BATCH {i}")
                tqdm_train_loader.set_postfix(self.aggregate_train_metrics())
        self.save_train_metrics(epoch_num)
        self.refresh_train_metrics()

    def val_epoch(self, epoch_num: int = None):
        with tqdm(self.val_loader) as tqdm_val_loader:
            for i, batch in enumerate(tqdm_val_loader):
                self.val_step(batch)
                tqdm_val_loader.set_description(f"BATCH {i}")
                tqdm_val_loader.set_postfix(self.aggregate_val_metrics())
        self.save_val_metrics(epoch_num)
        self.refresh_val_metrics()

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pt")
        self.api.upload_file(
            path_in_repo=self.config.save_repo_dir + "/" + self.model_name + ".pt",
            path_or_fileobj="./model.pt",
            repo_type="model",
            repo_id=self.config.hf_repo,
            token=self.config.token,
        )
        self.api.upload_folder(
            folder_path=self.log_dir,
            path_in_repo=self.config.tensorboard_dir + "/" + self.model_name,
            repo_type="model",
            repo_id=self.config.hf_repo,
            token=self.config.token,
        )
        os.remove("./model.pt")

    def train(self):
        with trange(self.config.epochs) as tqdm_epoch:
            for epoch in tqdm_epoch:
                tqdm_epoch.set_description(f"EPOCH {epoch}")
                self.model.train()
                self.train_epoch(epoch)
                self.model.eval()
                self.val_epoch(epoch)
                self.writer.flush()
        self.save_model()