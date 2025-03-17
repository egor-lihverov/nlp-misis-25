import torch
import torch.nn as nn
from torch.optim import SGD
from torchmetrics import AUROC, MeanMetric
from aim import Run
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 0.0),
        )

        self.aim_run = Run(experiment=self.config["experiment_name"])
        hparams = {
            "learning_rate": self.config["lr"],
            "hidden_size": self.config["model_params"]["hidden_size"],
            "batch_size": self.config["batch_size"],
            "n_epochs": self.config["n_epochs"],
            "weight_decay": self.config.get("weight_decay", 0),
            "dropout_p": self.config["model_params"].get("p", 0),
        }
        self.aim_run["hparams"] = hparams

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = {
            "loss": {"train": [], "val": []},
            "auc": {"train": [], "val": []},
        }

    def _train_epoch(self):
        self.model.train()
        train_loss = MeanMetric().to(self.device)
        train_auc = AUROC(task="binary").to(self.device)

        for batch in self.train_loader:
            X, y = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss)
            train_auc.update(torch.sigmoid(outputs), y)

        return train_loss.compute().item(), train_auc.compute().item()

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        val_loss = MeanMetric().to(self.device)
        val_auc = AUROC(task="binary").to(self.device)

        for batch in self.val_loader:
            X, y = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            val_loss.update(loss)
            val_auc.update(torch.sigmoid(outputs), y)

        return val_loss.compute().item(), val_auc.compute().item()

    def train(self):
        for epoch in tqdm(
            range(self.config["n_epochs"]), desc=self.config["experiment_name"]
        ):
            train_loss, train_auc = self._train_epoch()
            val_loss, val_auc = self._validate()

            self.history["loss"]["train"].append(train_loss)
            self.history["auc"]["train"].append(train_auc)
            self.history["loss"]["val"].append(val_loss)
            self.history["auc"]["val"].append(val_auc)

            # Логгирование метрик в Aim
            self.aim_run.track(train_loss, name="loss/train", step=epoch)
            self.aim_run.track(train_auc, name="roc-auc/train", step=epoch)
            self.aim_run.track(val_loss, name="loss/val", step=epoch)
            self.aim_run.track(val_auc, name="roc-auc/val", step=epoch)

            print(f"Epoch {epoch+1}/{self.config['n_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        return self.history
