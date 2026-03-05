import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as F_audio

from ..utils.common import make_padding_mask


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, E_dim, num_classes, loss_type='cosface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        in_features = E_dim
        out_features = num_classes
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps
        print('Initialised AngularPenaltySMLoss m=%.3f s=%.3f'%(self.m, self.s))
        print('Embedding dim is {}, number of classes is {}'.format(E_dim, num_classes))
    def forward(self, x, labels=None):
        '''
        input shape (N, in_features)
        '''
        # assert len(x) == len(labels)
        # assert torch.min(labels) >= 0
        # assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if labels==None:
            return wf

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L), wf



class AdditiveAngularMarginLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float,
        margin: float,
    ):
        super().__init__()

        self.scale = scale
        self.margin = margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.W = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_normal_(self.W)

    def forward(self, embeds: torch.Tensor, targets: torch.Tensor):
        x_norm = F.normalize(embeds, p=2.0, dim=1)
        w_norm = F.normalize(self.W, p=2.0, dim=1)

        cosine = torch.mm(x_norm, w_norm).clamp(-1.0, 1.0)
        sine = (1.0 - cosine.pow(2)).clamp(1e-9).sqrt()

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        masks = embeds.new_zeros(cosine.size())
        masks.scatter_(1, targets.view(-1, 1), 1.0)

        logits = masks * phi + (1.0 - masks) * cosine
        logits = self.scale * logits

        loss = F.cross_entropy(logits, targets)

        return loss


class ContrastiveLearningLoss(nn.modules.loss._Loss):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        samples: torch.Tensor,
    ):
        anchors = F.normalize(anchors, p=2.0, dim=1)
        samples = F.normalize(samples, p=2.0, dim=1)

        logits = torch.mm(anchors, samples.T)
        logits = (logits / self.temperature).exp()

        loss = logits.diag() / logits.sum(1)
        loss = -1.0 * loss.add(1e-9).log().mean()

        return loss


class IndependentBinaryClassificationLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: Optional[float] = 32.0,
        margin: Optional[float] = 0.15,
        lanbuda: Optional[float] = 0.7,
        t: Optional[int] = 3,
    ):
        super().__init__()

        self.scale = scale
        self.margin = margin
        self.lanbuda = lanbuda
        self.t = t

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = 1 + math.cos(math.pi - margin)

        self.W = nn.Parameter(torch.empty(embedding_dim, num_classes))
        nn.init.xavier_normal_(self.W)
        self.B = nn.Parameter(torch.empty(1))
        nn.init.zeros_(self.B)

    def forward(self, embeds: torch.Tensor, labels: torch.Tensor):
        x_norm = F.normalize(embeds, p=2.0, dim=1)
        w_norm = F.normalize(self.W, p=2.0, dim=1)

        cosine = torch.mm(x_norm, w_norm).clamp(-1.0, 1.0)
        sine = (1.0 - cosine.pow(2)).clamp(1e-9).sqrt()

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        cos_m_theta_p = self.scale * self._map(phi, self.t) + self.B

        phi = cosine * self.cos_m + sine * self.sin_m
        cos_m_theta_n = self.scale * self._map(phi, self.t) + self.B

        cos_p_theta = self.lanbuda * (1 + (-1.0 * cos_m_theta_p).exp()).log()
        cos_n_theta = (1 - self.lanbuda) * (1 + cos_m_theta_n.exp()).log()

        mask = embeds.new_zeros(cosine.size())
        mask.scatter_(1, labels.view(-1, 1), 1.0)

        loss = mask * cos_p_theta + (1 - mask) * cos_n_theta
        loss = loss.sum(1).mean()

        return loss

    def _map(self, z: torch.Tensor, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz


class RandomQuantizationLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        feature_dim: int,
        encoder_dim: int,
        embedding_dim: int,
        num_embeddings: int,
        num_codebooks: int,
    ):
        super().__init__()

        self.normalize = nn.LayerNorm(feature_dim, elementwise_affine=False)

        projection = torch.empty(feature_dim, num_codebooks * embedding_dim)
        nn.init.xavier_uniform_(projection)
        self.register_buffer("projection", projection)

        embeddings = torch.empty(num_embeddings, num_codebooks, embedding_dim)
        nn.init.normal_(embeddings)
        self.register_buffer("embeddings", embeddings)

        self.weight = nn.Parameter(
            torch.empty(num_codebooks, encoder_dim, num_embeddings)
        )
        nn.init.trunc_normal_(self.weight, std=0.02)

        self.bias = nn.Parameter(torch.empty(num_codebooks, num_embeddings))
        nn.init.zeros_(self.bias)

    def forward(
        self, xs: torch.Tensor, x_lens: torch.Tensor, ys: torch.Tensor
    ) -> torch.Tensor:
        ys = torch.matmul(self.normalize(ys), self.projection)

        b, t, c = ys.size()
        ys = ys.reshape(b * t, c)

        ys = self._quantize_vector(ys, self.embeddings)
        ys = ys.reshape(b, t, -1)

        xs = torch.matmul(xs[:, None, :, :], self.weight[None, :, :, :])
        xs = xs + self.bias[None, :, None, :]

        masks = make_padding_mask(x_lens, xs.size(2))
        loss = self._compute_loss(xs, ys, masks)

        return loss

    def _quantize_vector(
        self, latent: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        b, d = latent.size()
        _, g, _ = codebook.size()

        latent = latent.reshape(b, g, d // g)
        _codebook = codebook.permute(2, 1, 0)

        distance = (
            torch.sum(latent**2, dim=2, keepdim=True)
            - 2 * torch.einsum("bgd,cgd->bgc", latent, codebook)
            + torch.sum(_codebook**2, dim=0, keepdim=True)
        )

        code_ids = torch.argmin(distance, dim=2)

        return code_ids

    def _compute_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        xs = -1.0 * xs.log_softmax(dim=-1).transpose(1, 2)
        ys = ys[:, :, :, None]

        outs = xs.gather(3, ys).squeeze(3)
        outs = outs * masks[:, :, None]

        loss = outs.sum() / (masks.sum() * xs.size(2))

        return loss


class SequenceToSequenceLoss(nn.modules.loss._Loss):
    def __init__(
        self,
        ctc_weight: float = 1.0,
        rnnt_weight: float = 1.0,
    ):
        super(SequenceToSequenceLoss, self).__init__()
        self.blank_label = 0
        self.ctc_weight = ctc_weight
        self.rnnt_weight = rnnt_weight

    def forward(
        self,
        ctc_logits: torch.Tensor,
        rnnt_logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        ctc_loss = F.ctc_loss(
            log_probs=ctc_logits.transpose(0, 1),
            targets=targets,
            input_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=self.blank_label,
            zero_infinity=True,
        )

        rnnt_loss = F_audio.rnnt_loss(
            logits=rnnt_logits,
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=self.blank_label,
        )

        loss = self.ctc_weight * ctc_loss + self.rnnt_weight * rnnt_loss

        return loss, ctc_loss, rnnt_loss


class LeastSquaresGenerativeLoss(nn.modules.loss._Loss):
    def forward(self, disc_outs: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for dg in disc_outs:
            loss += torch.mean((1 - dg) ** 2)

        loss = loss / len(disc_outs)

        return loss


class LeastSquaresAdversarialLoss(nn.modules.loss._Loss):
    def forward(
        self,
        disc_outs: List[torch.Tensor],
        disc_tgts: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for dg, dr in zip(disc_outs, disc_tgts):
            loss += torch.mean((1 - dr) ** 2) + torch.mean(dg**2)

        loss = loss / len(disc_tgts)

        return loss


class STFTLoss(nn.modules.loss._Loss):
    def __init__(self, n_fft: int, win_length: int, hop_length: int):
        super(STFTLoss, self).__init__()
        self.transform = T.Spectrogram(n_fft, win_length, hop_length, power=1)

    def forward(
        self,
        audio_outs: torch.Tensor,
        audio_tgts: torch.Tensor,
        audio_masks: torch.Tensor,
    ) -> torch.Tensor:
        mel_outs = self.transform(audio_outs)
        mel_tgts = self.transform(audio_tgts)

        masks = F.interpolate(audio_masks.float(), mel_tgts.size(2))
        masks = masks.bool().expand_as(mel_tgts)

        sc_loss = self.spectral_convergence_loss(mel_outs, mel_tgts, masks)
        mag_loss = self.log_stft_magnitude_loss(mel_outs, mel_tgts, masks)

        loss = sc_loss + mag_loss

        return loss

    def spectral_convergence_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        numerator = ((ys - xs) * masks).norm(p="fro")
        denominator = (ys * masks).norm(p="fro")

        loss = numerator / (denominator + 1e-9)

        return loss

    def log_stft_magnitude_loss(
        self, xs: torch.Tensor, ys: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        xs = xs.add(1e-9).log()
        ys = ys.add(1e-9).log()

        loss = F.l1_loss(xs, ys, reduction="none")
        loss = (loss * masks).sum() / masks.sum()

        return loss


class MultiResolutionSTFTLoss(nn.modules.loss._Loss):
    def __init__(self, resolutions: List[Tuple[int, int, int, int, int]]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.losses = nn.ModuleList([STFTLoss(*res) for res in resolutions])

    def forward(
        self,
        audio_outs: torch.Tensor,
        audio_tgts: torch.Tensor,
        audio_lens: torch.Tensor,
    ) -> torch.Tensor:
        audio_outs = audio_outs.squeeze(1)
        audio_tgts = audio_tgts.squeeze(1)

        audio_masks = make_padding_mask(audio_lens, audio_outs.size(1))
        audio_masks = audio_masks[:, None, :]

        loss = 0.0
        for loss_fn in self.losses:
            loss += loss_fn(audio_outs, audio_tgts, audio_masks)

        loss = loss / len(self.losses)

        return loss


class TemporalPredictionLoss(nn.modules.loss._Loss):
    def forward(
        self, outs: torch.Tensor, tgts: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        tgts = tgts.add(1e-9).log()
        masks = make_padding_mask(lens, tgts.size(1))

        loss = F.smooth_l1_loss(outs, tgts, reduction="none")
        loss = (loss * masks).sum() / masks.sum()

        return loss


class AlignmentAttentionLoss(nn.modules.loss._Loss):
    def forward(
        self,
        soft_attns: torch.Tensor,
        hard_attns: torch.Tensor,
        text_lens: torch.Tensor,
        feat_lens: torch.Tensor,
    ) -> torch.Tensor:
        soft_attns = soft_attns.transpose(0, 1)
        hard_attns = hard_attns.transpose(0, 1)

        device = soft_attns.device
        batch_size = soft_attns.size(1)
        max_text_length = soft_attns.size(2)

        targets = torch.arange(1, max_text_length + 1, device=device)
        targets = targets.repeat(batch_size, 1)

        log_probs = F.pad(soft_attns, (1, 0, 0, 0, 0, 0), value=-1)
        log_probs = log_probs.log_softmax(2)

        loss = F.ctc_loss(
            log_probs,
            targets,
            feat_lens,
            text_lens,
            zero_infinity=True,
        )

        # bin_loss = soft_attns[hard_attns == 1].sum()
        # bin_loss = -bin_loss / hard_attns.sum()

        # loss = forward_sum_loss + bin_loss

        return loss


class SupervisedContrastiveLearningLoss(nn.modules.loss._Loss):
    def __init__(self, alpha: float, smoothing: float, temperature: float):
        super().__init__()
        self.alpha = alpha
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(
        self, embeds: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        cl_loss = self._contrastive_learning_loss(embeds, labels)
        ce_loss = self._cross_entropy_loss(logits, labels)
        loss = (1 - self.alpha) * ce_loss + self.alpha * cl_loss
        return loss, cl_loss, ce_loss

    def _contrastive_learning_loss(
        self, embeds: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        embeds = F.normalize(embeds, p=2.0, dim=1)
        embeds = torch.matmul(embeds, embeds.T) / self.temperature

        labels = labels[None, :] == labels[:, None]
        masks = torch.zeros_like(labels).fill_diagonal_(True)

        embeds = embeds[~masks].view(embeds.shape[0], -1)
        labels = labels[~masks].view(labels.shape[0], -1)

        numerator = embeds.masked_fill(~labels, -1e3).exp()
        denominator = embeds.exp().sum(dim=1)

        loss = numerator.sum(1) / denominator
        loss = -1.0 * loss.add(1e-9).log().mean()

        return loss

    def _cross_entropy_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        loss = F.cross_entropy(logits, labels, label_smoothing=self.smoothing)
        return loss


class MelSpectrogramLoss(nn.modules.loss._Loss):
    def forward(
        self,
        mel_outs: torch.Tensor,
        mel_tgts: torch.Tensor,
        mel_lens: torch.Tensor,
    ):
        mask = make_padding_mask(mel_lens, mel_tgts.size(1))
        mask = mask[:, :, None].expand_as(mel_tgts)

        loss = F.mse_loss(mel_outs, mel_tgts, reduction="none")
        loss = (loss * mask).sum() / mask.sum()

        return loss
