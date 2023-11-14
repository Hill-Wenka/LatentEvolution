import os
import sys

from omegaconf import OmegaConf

from framework.utils.config.config_utils import merge_config, config_to_namespace, config_to_dict
from framework.utils.log.log_utils import log_args

framework_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(framework_path)[0]

if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(framework_path)
    print('========== add root_path and framework_path to sys.path ==========')
    print('root_path:', root_path)
    print('framework_path:', framework_path)
    print('==================================================================')

'''Set the PYTHON_PATH and store common constants (paths)'''

paths = OmegaConf.create()
# paths['dataset'] = 'E:\\Data\\'
paths['path_dataset'] = '/home/hew/python/data/'

paths['root'] = os.path.join(root_path, '')
paths['data'] = os.path.join(root_path, 'data', '')
paths['script'] = os.path.join(root_path, 'script', '')
paths['temp'] = os.path.join(root_path, 'temp', '')
paths['cache'] = os.path.join(root_path, 'cache', '')

paths['framework'] = os.path.join(framework_path, '')
paths['config'] = os.path.join(framework_path, 'config', '')
paths['module'] = os.path.join(framework_path, 'module', '')
paths['utils'] = os.path.join(framework_path, 'utils', '')
