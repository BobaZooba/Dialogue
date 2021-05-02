from dataclasses import dataclass
from typing import Dict, Optional
from typing import List, Sequence, Union, Tuple

import torch
from torch import Tensor


@dataclass
class Types:

    phrase: str = 'phrase'
    response: str = 'response'
    context: str = 'context'
    candidates: str = 'candidates'
    target: str = 'target'


TYPES = Types()

RawTextData = List[Dict[str, Union[str, List[str]]]]
TextBatch = Dict[str, Union[str, int, float, Sequence[str]]]
Batch = Dict[str, Tensor]
SequenceBatch = Sequence[Batch]
