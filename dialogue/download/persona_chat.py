import os
import logging
import sys
import requests
import random
from dialogue import io, utils

PERSONA_CHAT_URL = 'https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json'
BAD_INPUT_PHRASES = [
    '__ SILENCE __'
]


def download_data():

    request = requests.get(url=PERSONA_CHAT_URL)

    data = request.json()

    train_data, valid_data = data['train'], data['valid']

    return train_data, valid_data


def get_dialogues(data):
    dialogues = list()

    for sample in data:

        dialogue = sample['utterances'][-1]['history'] + [sample['utterances'][-1]['candidates'][-1]]

        if dialogue[0] == BAD_INPUT_PHRASES:
            dialogue = dialogue[1:]

        if len(dialogue) < 2:
            continue

        dialogues.append(dialogue)

    return dialogues


def dialogues2bi_encoder_samples(dialogues):

    unique_phrases = set()
    data = list()

    for dialogue in dialogues:

        for i_phrase in range(len(dialogue) - 1):

            sample = {
                io.TYPES.phrase: dialogue[i_phrase],
                io.TYPES.response: dialogue[i_phrase + 1],
                io.TYPES.context: list() if i_phrase == 0 else dialogue[:i_phrase]
            }

            data.append(sample)

        unique_phrases.update(dialogue)

    unique_phrases = list(unique_phrases)

    return data, unique_phrases


def dialogues2cross_encoder_samples(dialogues, unique_phrases, n_negative: int = 4):

    data = list()

    for dialogue in dialogues:

        for i_phrase in range(len(dialogue) - 1):

            sample = {
                io.TYPES.phrase: dialogue[i_phrase],
                io.TYPES.response: dialogue[i_phrase + 1],
                io.TYPES.target: 1,
                io.TYPES.context: list() if i_phrase == 0 else dialogue[:i_phrase]
            }

            data.append(sample)

            for _ in range(n_negative):

                sample = {
                    io.TYPES.phrase: dialogue[i_phrase],
                    io.TYPES.response: random.choice(unique_phrases),
                    io.TYPES.target: 0,
                    io.TYPES.context: list() if i_phrase == 0 else dialogue[:i_phrase]
                }

                data.append(sample)

    return data


def get_valid_samples(raw_data):
    data = list()

    for sample in raw_data:
        for turn in sample['utterances']:
            if turn['history'][-1] not in BAD_INPUT_PHRASES:
                sample = {
                    io.TYPES.phrase: turn['history'][-1],
                    io.TYPES.response: turn['candidates'][-1],
                    io.TYPES.context: turn['history'][:-1],
                    io.TYPES.candidates: turn['candidates'][:-1],
                }
                data.append(sample)

    return data


def run(data_path: str, n_negative: int = 4):

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger(os.path.basename(__file__))

    logger.info('Start Persona Chat Downloading')
    train_raw_data, valid_raw_data = download_data()
    logger.info('Data Downloaded')

    train_dialogues = get_dialogues(train_raw_data)
    logger.info('Train Dialogues Parsed')

    bi_encoder_train_data, unique_phrases = dialogues2bi_encoder_samples(train_dialogues)
    logger.info('Bi Encoder Train Samples Parsed')

    cross_encoder_train_data = dialogues2cross_encoder_samples(train_dialogues,
                                                               unique_phrases=unique_phrases,
                                                               n_negative=n_negative)
    logger.info('Cross Encoder Train Samples Parsed')

    valid_data = get_valid_samples(valid_raw_data)
    logger.info('Valid Samples Parsed')

    random.shuffle(bi_encoder_train_data)
    random.shuffle(cross_encoder_train_data)
    logger.info('Data Shuffled')

    utils.save_jsonl(file_path=os.path.join(data_path, 'bi_encoder_train.jsonl'), data=bi_encoder_train_data)
    logger.info('Bi Encoder Train Saved')

    utils.save_jsonl(file_path=os.path.join(data_path, 'cross_encoder_train.jsonl'), data=cross_encoder_train_data)
    logger.info('Cross Encoder Train Saved')

    utils.save_jsonl(file_path=os.path.join(data_path, 'valid.jsonl'), data=valid_data)
    logger.info('Valid Saved')
