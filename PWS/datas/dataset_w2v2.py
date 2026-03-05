import random
from collections import defaultdict
from typing import Tuple, List, Union, Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pesq import pesq
import json
import torch.nn.functional as F

import torchaudio.transforms as T
from transformers import Wav2Vec2Processor

emo2int={"neu":0, "ang":1, "hap":2, "sad":3}
int2emo={0:"neu", 1:"ang", 2:"hap", 3:"sad"}



def get_dataset(dataset_filepath):
    if isinstance(dataset_filepath, str):
        dataset_filepath = [dataset_filepath]

    dataset = []
    for filepath in dataset_filepath:
        with open(filepath) as f:
            lines = f.readlines()[1:]
            data=[]
            for line in lines:
                data.append({"audio_filepath": line.split(",")[0] , "label": emo2int[line.split("\n")[0].split(",")[1]]     }       )
        dataset.extend(data)

    return dataset

class EmoDataset(Dataset):
    def __init__(
        self,
        config,
        isTrain=False
    ):
        super().__init__()

        filepaths = config['filepaths']

        self.dataset = get_dataset(config['filepaths'])

        self.processor = Wav2Vec2Processor.from_pretrained(config["modelpath"])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio_filepath = data["audio_filepath"]
        label = data["label"]
        speech, sample_rate = torchaudio.load(audio_filepath)
        speech = speech.mean(dim=0, keepdim=True)
        if speech.shape[1] < 2 *sample_rate:
            speech = torch.cat( (speech, torch.zeros((1,  2 *sample_rate- speech.shape[1]))), dim=-1 )
        if sample_rate != 16000:
            speech = T.Resample(sample_rate, 16000)(speech)
            sample_rate = 16000

        audio = self.processor(speech, return_tensors='pt',sampling_rate=sample_rate).input_values.squeeze()


        return audio, label

    def __len__(self) -> int:
        return len(self.dataset)






class EmoDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super(EmoDataModule, self).__init__()
        self.train_config = config["train_ds"]
        self.val_config = config["val_ds"]
        self.transform_config = config.get("transformation", None)

    def setup(self, stage: Optional[str] = None):
        self.train_ds = EmoDataset(self.train_config, True)
        self.val_ds = EmoDataset(self.val_config)

    def train_dataloader(self) -> DataLoader:
        data_loader_config = self.train_config["data_loader"]
        train_dl = DataLoader(
            self.train_ds,
            collate_fn=self.collate_data,
            **data_loader_config,
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        data_loader_config = self.val_config["data_loader"]
        val_dl = DataLoader(
            self.val_ds,
            collate_fn=self.collate_data,
            **data_loader_config,
        )
        return val_dl

    @staticmethod
    def collate_data(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

        features = [batch[0] for batch in sorted_batch]
        feature_lengths = [len(feat) for feat in features]
        features = pad_sequence(features, batch_first=True)
        feature_lengths = torch.tensor(feature_lengths, dtype=torch.int)

        audio_mask = torch.arange(features.shape[1])[None, :] < feature_lengths[:, None]

        targets = [batch[1] for batch in sorted_batch]
        targets = torch.tensor(targets).type(torch.LongTensor)

        return features,  audio_mask , targets



