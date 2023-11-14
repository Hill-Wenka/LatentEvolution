from omegaconf import OmegaConf
from argparse import Namespace


def config_to_namespace(args):
    # 将config从OmegaConf转换为Namespace
    return Namespace(**OmegaConf.to_object(args))


def config_to_dict(args):
    # 将config从OmegaConf转换为dict
    return OmegaConf.to_object(args)


def load_args(path):
    '''
    从checkpoints的hparams文件中加载保存好的参数设置
    :param path: hparams的文件路径
    :return: args: Namespace/dict
    '''
    args = OmegaConf.load(path)
    try:
        args = args.args
    except:
        pass
    return args


def nest_dict(flat):
    '''
    将字典flat中包含"."的key进行拆分，创建嵌套字典
    '''
    
    def _nest_dict_rec(k, v, out):
        k, *rest = k.split('.', 1)
        if rest:
            _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
        else:
            out[k] = v
    
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result)
    return result


def merge_config(config, update_list):
    '''
    更新config, 统一处理dict类型，yaml文件路径和命令行参数
    '''
    for x in update_list:
        if type(x) == dict:
            config = OmegaConf.merge(config, x)
        elif type(x) == str:
            if x.endswith('.yaml') and '=' not in x:
                config = OmegaConf.merge(config, OmegaConf.load(x))
            else:
                config = OmegaConf.merge(config, OmegaConf.from_dotlist([x]))
        else:
            raise RuntimeError(f'There is a illegal in param update_list: {update_list}')
    return config
