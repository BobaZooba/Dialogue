import json
from dialogue import io
import importlib
from typing import Tuple, Any, Optional
from typing import Union, Sequence, List, Dict

from tqdm import tqdm
import json
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from dialogue.nn.common import TextInput, Normalization


def read_jsonl(file_path: str) -> io.RawTextData:

    data = list()

    with open(file_path) as file_object:
        for line in file_object:
            data.append(json.loads(line.strip()))

    return data


def save_jsonl(file_path: str, data: io.RawTextData):

    with open(file_path, 'w') as file_object:
        for sample in data:
            file_object.write(json.dumps(sample) + '\n')


def generator_jsonl(file_path: str, max_samples: int = -1, verbose: bool = False) -> List[Dict[str, Any]]:

    n_samples = 0

    progress_bar = tqdm(total=max_samples, desc='Reading', disable=not verbose) \
        if max_samples > 0 else tqdm(desc='Reading', disable=not verbose)

    with open(file=file_path) as file_object:
        for line in file_object:
            sample = json.loads(line.strip())

            progress_bar.update()

            n_samples += 1

            if 0 < max_samples == n_samples:
                break

            yield sample

    progress_bar.close()


def parse_prefix(prefix: Optional[str] = None) -> str:
    if prefix and prefix[-1] != '_':
        prefix += '_'
    elif prefix is None:
        prefix = ''
    return prefix


def parse_postftix(postfix: Optional[str] = None) -> str:
    if postfix and postfix[0] != '_':
        postfix = '_' + postfix
    elif postfix is None:
        postfix = ''
    return postfix


def convert_to_torch_tensor(vectors: Union[np.ndarray, Tensor]):
    if not isinstance(vectors, Tensor):
        vectors = torch.tensor(vectors)
    return vectors


def normalize_embeddings(embeddings: Union[Tensor, Sequence[Tensor]]) -> Union[Tensor, Sequence[Tensor]]:

    if isinstance(embeddings, Tensor):
        return F.normalize(embeddings).detach()
    else:
        return tuple(F.normalize(part).detach() for part in embeddings)


def get_non_eye_matrix(matrix: Tensor) -> Tensor:

    batch_size = matrix.size(0)

    mask = torch.eye(batch_size).bool().to(matrix.device)

    matrix = matrix[~mask]
    matrix = matrix.view(batch_size, batch_size - 1)

    return matrix


def get_triu_matrix(matrix: Tensor) -> Tensor:
    mask = torch.triu(torch.ones_like(matrix)).bool().to(matrix.device)
    return matrix[mask]


def get_random_indices(batch_size: int, non_eye: bool = True) -> Tensor:

    indices_matrix = torch.arange(batch_size).unsqueeze(0).repeat(batch_size, 1)

    if non_eye:
        indices_matrix = get_non_eye_matrix(indices_matrix)

    indices_matrix = torch.stack([tensor[torch.randperm(batch_size - 1)] for tensor in indices_matrix])

    return indices_matrix


def import_object(module_path: str, object_name: str) -> Any:
    module = importlib.import_module(module_path)
    if not hasattr(module, object_name):
        raise AttributeError(f'Object `{object_name}` cannot be loaded from `{module_path}`.')
    return getattr(module, object_name)


def import_object_from_path(object_path: str, default_object_path: str = '') -> Any:
    object_path_list = object_path.rsplit('.', 1)
    module_path = object_path_list.pop(0) if len(object_path_list) > 1 else default_object_path
    object_name = object_path_list[0]
    return import_object(module_path=module_path, object_name=object_name)


def load_object(config: DictConfig) -> Any:

    _class = import_object_from_path(object_path=config.class_path)

    if not config.parameters:
        _object = _class()
    else:
        _object = _class(**config.parameters)

    return _object


def load_dual_layers(layer_config: DictConfig,
                     shared: bool) -> Tuple[nn.Module, nn.Module]:

    context_layer = load_object(layer_config)
    if shared:
        response_layer = context_layer
    else:
        response_layer = load_object(config=layer_config)

    return context_layer, response_layer


def model_assembly(backbone: nn.Module,
                   aggregation: nn.Module,
                   head: nn.Module,
                   embedding: Optional[nn.Module] = None,
                   tensor_input: bool = True,
                   add_normalization: bool = True,
                   pad_index: int = 0) -> nn.Module:

    model = nn.Sequential()

    if tensor_input:
        model.add_module('input', TextInput(pad_index=pad_index))

    if embedding is not None:
        model.add_module('embedding', embedding)

    model.add_module('backbone', backbone)
    model.add_module('aggregation', aggregation)
    model.add_module('head', head)

    if add_normalization:
        model.add_module('normalization', Normalization())

    return model

