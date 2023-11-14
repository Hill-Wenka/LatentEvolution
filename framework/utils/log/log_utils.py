import json
import argparse
import omegaconf
from omegaconf import OmegaConf


def log_args(args, prefix='args'):
    '''
    格式化输出配置参数args (dict/Namespace/omegaconf.dictconfig.DictConfig)
    :param args: 配置参数dict, Namespace或omegaconf.dictconfig.DictConfig
    :param prefix: 打印的前缀字符串
    :return: None
    '''
    if type(args) == dict:
        pass
    elif type(args) == omegaconf.dictconfig.DictConfig:
        args = OmegaConf.to_object(args)
    elif type(args) == argparse.Namespace:
        args = vars(args)
    else:
        raise RuntimeError(f'Param args is illegal type: {args}')
    print(f'[{prefix}]:')
    print(json.dumps(args, indent=4, ensure_ascii=False))  # 缩进4空格，中文字符不转义成Unicode


def log(variable, prefix, line_break=False):
    line_break = '\n' if line_break else ' '
    string = f'{prefix}:{line_break}{variable}'
    print(string)


if __name__ == '__main__':
    args = {'lr': 1e-3, 'batch_size': 32}
    log_args(args)

    variable = 123
    log(variable, 'variable', False)
    log(variable, 'variable', True)
