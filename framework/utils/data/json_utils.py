import json

import numpy as np
import omegaconf


def json2list(param):
    '''
    根据param的数据类型进行相应的转换，最终都转换为list
    :param param: 参数
    :return: param: list
    '''
    if isinstance(param, str):
        param = json.loads(param)
    elif isinstance(param, int):
        param = [param]
    elif isinstance(param, (list, omegaconf.listconfig.ListConfig)):
        pass
    else:
        raise RuntimeError(f'Parameter has illegal type: [{type(param)}]')
    return param


def params2list(*params):
    '''
    将json字符串参数转换为list，一般用在torch.nn模型中解析传入的params
    :param params: 参数列表
    :return: params: list
    '''
    return [json2list(param) for param in params]


def obj2str(obj):
    '''
    将对象转换为字符串
    :param obj: 对象
    :return: obj: str
    '''
    if isinstance(obj, str):
        string = obj
    elif isinstance(obj, int):
        string = str(obj)
    elif isinstance(obj, (list, dict, omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig)):
        string = json.dumps(obj)
    elif isinstance(obj, np.ndarray):
        string = json.dumps(obj.tolist())
    else:
        raise RuntimeError(f'Parameter has illegal type: [{type(obj)}]')
    return string
