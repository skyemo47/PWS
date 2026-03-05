from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from ..modules.encoder import AcousticEncoder
from ..modules.pooling import Attentive_Pool

from torch.optim import AdamW
from ..optims.scheduler import NoamAnnealing
from ..modules.criterion import AngularPenaltySMLoss
from torch.nn.utils.rnn import pad_sequence
from ..modules.wavlm import WavLMWrapper
from scipy.stats import pearsonr
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as score
import fairseq
import torch.nn as nn
import numpy as np
class VEMO(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = WavLMWrapper(config.cp_path)

        self.pooling = Attentive_Pool(**config.pooling)

        # Optimizer config
        self.optimizer_config = config.optimizer
        self.scheduler_config = config.scheduler
        self.num_classes = 4
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.pooling.E_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, self.num_classes)
        )
        self.initial_k: int = config.initial_k
        self.final_k: int = 1
        self.k_decay_start: int = 20
        self.k_decay_end: int = 150
        self.validation_step_outputs = []
        self.alpha= config.alpha
        self.loss_type=config.loss_type


    def get_current_k(self) -> int:
        """
        Calculate current k value based on training progress.
        Uses linear decay from initial_k to final_k.

        Returns:
            Current k value (integer)
        """
        epoch = self.current_epoch

        if epoch < self.k_decay_start:
            return self.initial_k
        elif epoch >= self.k_decay_end:
            return self.final_k
        else:
            # Linear interpolation
            progress = (epoch - self.k_decay_start) / (self.k_decay_end - self.k_decay_start)
            k_float = self.initial_k + progress * (self.final_k - self.initial_k)
            return max(self.final_k, int(np.round(k_float)))

    def progressive_weak_supervision_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Calculate loss with progressive weak supervision.

        In early epochs: If true label is in top-k predictions,
                        consider it as learning correctly.
        In later epochs: Only top-1 must match (standard CE loss).

        Args:
            logits: Model predictions (batch_size, num_classes)
            labels: True labels (batch_size,)

        Returns:
            loss: Computed loss
            metrics: Dictionary of training metrics
        """
        batch_size = logits.size(0)
        k = self.get_current_k()

        # Get top-k predictions
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

        # Check if true label is in top-k
        labels_expanded = labels.unsqueeze(1).expand(-1, k)
        in_topk = (topk_indices == labels_expanded).any(dim=1)  # (batch_size,)

        if k > 1:
            # Weak supervision: Adjust targets for samples where label is in top-k
            # Create soft targets that distribute probability among top-k
            soft_targets = torch.zeros_like(probs)

            for i in range(batch_size):
                if in_topk[i]:
                    # If label in top-k, create soft target with label having highest prob
                    # and other top-k classes sharing remaining probability
                    soft_targets[i, labels[i]] = self.alpha # Main probability to true label
                    mask = torch.zeros(self.num_classes, dtype=torch.bool, device=logits.device)
                    mask[topk_indices[i]] = True
                    mask[labels[i]] = False  # Exclude true label
                    num_other = mask.sum()
                    if num_other > 0:
                        soft_targets[i, mask] = (1-self.alpha) / num_other
                else:
                    # Standard one-hot for samples where label is not in top-k
                    soft_targets[i, labels[i]] = 1.0

            # KL divergence loss with soft targets
            log_probs = F.log_softmax(logits, dim=-1)
            if self.loss_type =="KL":
                loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
            else:
                loss = F.cross_entropy(logits, soft_targets)

        else:
            # k=1: Standard cross-entropy loss
            loss = F.cross_entropy(logits, labels)

        # Calculate accuracy metrics
        pred_labels = logits.argmax(dim=-1)
        top1_acc = (pred_labels == labels).float().mean()
        topk_acc = in_topk.float().mean()

        metrics = {
            'k': float(k),
            'top1_acc': top1_acc.item(),
            f'top{k}_acc': topk_acc.item(),
            'in_topk_ratio': in_topk.float().mean().item()
        }

        return loss, metrics


    def forward(self, batch):

        a_lens = [len(feat) for feat in batch]

        a_features = pad_sequence(batch, batch_first=True)
        out = self.shared_step(a_features, a_lens)
        _, predicted = torch.max(out, -1)
        return predicted


    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):

        xs, x_lens, ys = batch

        logits = self.shared_step(xs, None)

        # Progressive weak supervision loss
        loss, metrics = self.progressive_weak_supervision_loss(logits, ys)

        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_top1_acc', metrics['top1_acc'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('current_k', metrics['k'], prog_bar=True, on_step=False, on_epoch=True)
        for key, value in metrics.items():
            if key != 'top1_acc':
                self.log(f'train_{key}', value, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        xs, x_lens, labels = batch

        # print(labels.shape, labels.dtype)
        logits = self.shared_step(xs, x_lens)
        # Standard cross-entropy for validation
        loss = F.cross_entropy(logits, labels)

        # Accuracy
        pred_labels = logits.argmax(dim=-1)
        acc = (pred_labels == labels).float().mean()

        # Top-k accuracy for analysis
        k = self.get_current_k()
        if k > 1:
            topk_probs, topk_indices = torch.topk(logits, k=k, dim=-1)
            labels_expanded = labels.unsqueeze(1).expand(-1, k)
            topk_acc = (topk_indices == labels_expanded).any(dim=1).float().mean()
        else:
            topk_acc = acc

        self.log("vtopk_acc", topk_acc, prog_bar=True, sync_dist=True)
        self.log("v_loss", loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append((logits.detach().cpu(), labels.detach().cpu(), loss.detach().cpu()))


    def on_validation_epoch_end(self):
        preds, keys = [], []
        losses=0
        for output in self.validation_step_outputs:
            preds.append(output[0])
            keys.append(output[1])
            losses  = losses + output[2].item()
        losses = losses/len(self.validation_step_outputs)
        preds = torch.cat(preds, dim=0)
        keys = torch.cat(keys, 0)
        _, predicted = torch.max(preds, -1)
        c = (predicted == keys).squeeze().sum()
        acc = c/keys.shape[0]

        keys = keys.numpy()
        predicted = predicted.numpy()
        self.validation_step_outputs = []
        precision, recall, fscore, support = score(keys, predicted, zero_division=1)
        self.log("val_acc", acc, prog_bar=True)
        self.log("f1_score", fscore.mean(), prog_bar=True)
        self.log("val_loss", losses, prog_bar=True)
        self.log("ua", recall.mean(), prog_bar=True)


    def shared_step(
        self,
        xs: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        emb = self.encoder(xs)
        out = self.pooling(emb, None)
        logit = self.classifier(out)

        return logit



    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.optimizer_config)
        scheduler = NoamAnnealing(optimizer, **self.scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
