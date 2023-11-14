import os
import shutil


def is_path_exist(path):
    '''
    检查路径是否存在
    :param path: 检查的文件路径
    :return: is_exist: bool
    '''
    return os.path.exists(path)


def check_path(path, mkdir=True, log=True):
    '''
    检查路径所在文件夹是否存在, 如果路径不存在则自动新建
    :param path: 检查的文件路径
    :param mkdir: 是否自动创建文件夹
    :return: is_exist: bool
    '''
    dir = os.path.abspath(os.path.dirname(path)) if not os.path.isdir(path) else path
    is_exist = is_path_exist(dir)
    if mkdir and not is_exist:
        try:
            os.makedirs(dir, exist_ok=True)
            if log:
                print(f'The path does not exist, makedir: {dir}: Success')
        except Exception:
            raise RuntimeError(f'The path does not exist, makedir {dir}: Failed')
    return is_exist


def makedir(path):
    os.makedirs(path, exist_ok=True)


def walk_path(base):
    '''
    遍历base文件夹（目录），返回所有的路径组合（root, dir, file）
    :param base: 指定的目录
    :return: results: list
    '''
    results = []
    for root, dirs, files in os.walk(base):
        results.append([root, dirs, files])
    return results


def list_dirs(base, absolute=False):
    '''
    遍历base文件夹（目录），返回当前文件夹下的所有子文件夹
    :param base: 指定的目录
    :param absolute: 是否返回绝对路径
    :return: list
    '''
    if absolute:
        return [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    else:
        return [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]


def list_files(base, absolute=False):
    '''
    遍历base文件夹（目录），返回当前文件夹下的所有子文件
    :param base: 指定的目录
    :param absolute: 是否返回绝对路径
    :return: list
    '''
    if absolute:
        return [os.path.join(base, f) for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
    else:
        return [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]


def filter_dirs(path, string):
    '''
    递归遍历指定文件夹下的所有文件夹，筛选指定字符串的文件夹
    :param path: 指定的文件夹路径
    :param suffix: 指定的文件夹名称
    :return: list
    '''
    results = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if string in dir:
                results.append(os.path.join(root, dir))
    return results


def filter_files(path, suffix):
    '''
    递归遍历指定文件夹下的所有文件，筛选指定后缀的文件
    :param path: 指定的文件夹路径
    :param suffix: 指定的文件后缀
    :return: list
    '''
    return [file for file in list_files(path) if file.endswith(suffix)]


def remove_dirs(dirs, force=True):
    '''
    删除指定的文件夹
    :param dirs: 指定的文件夹路径
    :param force: 是否强制删除
    :return: None
    '''
    if isinstance(dirs, str):
        dirs = [dirs]
    for dir in dirs:
        if os.path.isdir(dir):
            if force:
                os.system(f'rm -rf {dir}')
            else:
                os.rmdir(dir)
        else:
            print(f'Warning: {dir} is not a directory')


def remove_files(files):
    '''
    删除指定的文件
    :param files: 指定的文件路径
    :return: None
    '''
    if isinstance(files, str):
        files = [files]
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
        else:
            print(f'Warning: {file} is not a file')


def rename_file(file, new_name):
    '''
    批量重命名文件
    :param files:
    :param new_name:
    :return:
    '''
    if os.path.isfile(file):
        os.rename(file, new_name)
    else:
        print(f'Warning: {file} is not a file')


def copy_file(src_path, target_path):
    is_exist = is_path_exist(src_path)
    if is_exist:
        check_path(target_path)
        shutil.copy(src_path, target_path)
        print(f'Copy {src_path} to {target_path}: Success')
    else:
        print(f'Warning: {src_path} is not a file')


def get_basename(path, suffix=False):
    '''
    获取文件名
    :param path: 文件路径
    :return: 文件名
    '''
    return os.path.basename(path) if suffix else os.path.splitext(os.path.basename(path))[0]
