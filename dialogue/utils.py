import json
from dialogue import io


def load_jsonl(file_path: str) -> io.RawTextData:

    data = list()

    with open(file_path) as file_object:
        for line in file_object:
            data.append(json.loads(line.strip()))

    return data


def save_jsonl(file_path: str, data: io.RawTextData):

    with open(file_path, 'w') as file_object:
        for sample in data:
            file_object.write(json.dumps(sample) + '\n')
