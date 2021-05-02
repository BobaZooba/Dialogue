from dialogue.utils import read_jsonl
from torch.utils.data import Dataset, DataLoader
from dialogue import io
import torch
from dialogue.data import dataset, preparing, tokenization

from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, AutoModel
from torch import nn, Tensor


class HuggingFaceBertBackbone(nn.Module):

    def __init__(self, model_name: str):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, batch: io.Batch) -> io.Batch:

        batch[io.TYPES.backbone] = self.model(input_ids=batch[io.TYPES.sequence_indices],
                                              attention_mask=batch[io.TYPES.pad_mask]).last_hidden_state

        return batch
