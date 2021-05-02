import logging
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dialogue.download import persona_chat


@hydra.main(config_path='../conf/download/', config_name='config')
def main(config: DictConfig):

    os.environ['HYDRA_FULL_ERROR'] = '1'

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger(os.path.basename(__file__))

    yaml_config = OmegaConf.to_yaml(config)
    logger.info(yaml_config)

    os.makedirs(config.data_path, exist_ok=True)

    if config.data_type == 'persona_chat':
        persona_chat.run(data_path=config.data_path, n_negative=config.cross_encoder_n_negative)
    else:
        raise ValueError('data_type not exist')


if __name__ == '__main__':
    main()
