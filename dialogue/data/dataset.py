from dialogue.utils import read_jsonl
from torch.utils.data import Dataset, DataLoader
from dialogue import io
import torch

from transformers import AutoTokenizer, AutoModel


class ConversationalDataset(Dataset):

    def __init__(self, file_path):
        super().__init__()
        self.data = read_jsonl(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
