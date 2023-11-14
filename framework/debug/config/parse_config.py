import os

import omegaconf
from omegaconf import OmegaConf

from framework.utils.log.log_utils import log_args


def parse_config(update_dict=None) -> omegaconf.dictconfig.DictConfig:
    config = OmegaConf.create()
    print('os.sys.argv', os.sys.argv)
    for x in os.sys.argv[1:]:
        if x.endswith('.yaml') and '=' not in x:
            config = OmegaConf.merge(config, OmegaConf.load(x))
        else:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist([x]))
    if update_dict is not None:
        config = OmegaConf.merge(config, update_dict)
    return config


if __name__ == '__main__':
    update_dict = {
        'project': 'test',
        'data_args': {
            'dataframe': 'test'
        }
    }
    config = parse_config(update_dict)
    print('type(config)', type(config))
    print('config.project', config.project)
    print('config.data_args.dataframe', config.data_args.dataframe)
    print('config', config)
    log_args(config)
