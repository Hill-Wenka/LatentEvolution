import os

from omegaconf import OmegaConf

from framework import paths
from framework.utils.config.config_utils import merge_config, config_to_namespace, config_to_dict, load_args
from framework.utils.log.log_utils import log_args


def select_key(key, param, update_list):
    '''
    从update_list中寻找是否有指定“key”的更新值，如果有则更新，否则用param中key的值。
    统一处理dict类型，yaml文件路径和命令行参数。
    '''

    for x in update_list:
        if type(x) == dict:
            # 如果key在x中，则更新param
            if key in x:
                param = x[key]
        elif type(x) == str:
            if x.endswith('.yaml') and '=' not in x:
                # 从yaml文件中加载
                config = OmegaConf.load(x)
                # 如果key在config中，则更新param
                if key in config:
                    param = config[key]
            elif f'{key}=' in x:
                # 从命令行参数中加载
                param = x[len(f'{key}='):]
            else:
                pass
        else:
            print('[error] x', x)
            raise RuntimeError(f'There is a illegal args in param update_list: {update_list}')

    if not param.endswith('.yaml'):
        param = paths.config + f'{key}/{param}.yaml'
    return param


def parse_config(update_list=None):
    '''
    从yaml文件中加载默认配置，包括项目设置和模型超参数
    :param update_dict: 自定义的参数设置，如果不为None则将对应字段的最新指定值覆盖默认值
    :return: args: OmegaConf (omegaconf.dictconfig.DictConfig)
    '''
    if update_list is None:
        update_list = []

    config = OmegaConf.create()
    default_config_list = ['script/script.yaml', 'lightning/LitData.yaml', 'lightning/LitModel.yaml', 'others/IO.yaml', 'others/NNI.yaml']
    default_config_list = [paths.config + x for x in default_config_list]
    config = merge_config(config, default_config_list)
    config.data = select_key('dataset', config.dataset, update_list + os.sys.argv[1:])
    config.model = select_key('model', config.model, update_list + os.sys.argv[1:])
    config = merge_config(config, [config.data, config.model] + update_list + os.sys.argv[1:])
    return config


if __name__ == '__main__':
    update_dict = {
        'project': 'debug',
        'model': 'MLP',
        'data': {
            'dataframe': 'test.pt'
        },
        'hparams': {
            'activation': 'Sigmoid'
        }
    }
    args = parse_config([update_dict])
    print('===== format args =====')
    log_args(args)

    print('===== args in OmegaConf =====')
    print(type(args), args)

    args_dict = config_to_dict(args)
    print('===== args in Dict =====')
    print(type(args_dict), args_dict)

    args_namespace = config_to_namespace(args)
    print('===== args in Namespace =====')
    print(type(args_namespace), args_namespace)

    # resume hparams
    print(paths['framework'])
    # ckpt_args = paths['framework'] + 'environment/check/lightning_logs/old_version_0_backup/hparams.yaml'
    ckpt_args = paths['framework'] + 'debug/config/config_1.yaml'
    ckpt_args = load_args(ckpt_args)
    print('===== reuse checkpoint args in OmegaConf =====')
    print(type(ckpt_args), ckpt_args)
