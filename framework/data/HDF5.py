import h5py
import numpy as np

from framework.utils.data.dict_utils import merge_dicts
from framework.utils.data.pickle_utils import serialize, deserialize
from framework.utils.parallel.parallel_utils import asyn_parallel


def recursively_save_dict_contents(h5file, path, data_dict):
    for key, value in data_dict.items():  # iterate over keys, values in the dictionary
        if isinstance(value, dict):
            recursively_save_dict_contents(h5file, path + '/' + key, value)  # recursively save dictionary in this group
        else:
            if isinstance(value, (int, float, str, bool)):
                pass
            elif isinstance(value, str):
                value = value.encode()
            elif isinstance(value, (list, np.ndarray)):
                if len(value) > 0 and isinstance(value[0], str):
                    value = [serialize(value, binary=True)]
            else:
                value = [serialize(value, binary=True)]

            save_dataset(h5file, path + '/' + str(key), value)


def save_dataset(h5file, path, data):
    # print(f'if [{path}] exists', path in h5file)
    if path not in h5file:
        h5file.create_dataset(path, data=data)
    else:
        h5file[path][...] = data


def load_dataset(f, path):
    # recursively read all levels of data
    data = {}
    for key, value in f[path].items():
        if isinstance(value, h5py._hl.dataset.Dataset):
            data[key] = decode_object_data(value[...])
        elif isinstance(value, h5py._hl.group.Group):
            data[key] = load_dataset(f, path + '/' + key)
    return data


def parallel_load_func(h5_file, group):
    with h5py.File(h5_file, 'r') as f:
        data = load_dataset(f, group)
    return {group: data}


def decode_object_data(value):
    if len(value.shape) > 0:
        if isinstance(value, np.ndarray) and isinstance(value[0], bytes):
            value = deserialize(value[0], binary=True)  # value is an object
    else:
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode()  # value is a string
    return value


class HDF5:
    def __init__(self, h5_file, **kwargs):
        self.h5_file = h5_file

    def save(self, group, dataset, data):
        # create group if not exists else get group
        with h5py.File(self.h5_file, 'a') as f:
            if group not in f:
                f.create_group(f'/{group}')
            path = f'/{group}/{dataset}'

            if isinstance(data, (int, float, bool)):
                save_dataset(f, path, data)
            elif isinstance(data, str):
                save_dataset(f, path, data.encode())
            elif isinstance(data, (list, np.ndarray)):
                if len(data) > 0 and isinstance(data[0], str):
                    data = [serialize(data, binary=True)]
                save_dataset(f, path, data)
            elif isinstance(data, dict):
                recursively_save_dict_contents(f, path, data)
            else:
                data = [serialize(data, binary=True)]
                save_dataset(f, path, data)

    def load(self, groups=None):
        groups = self.groups if groups is None else groups
        hdf5_data = asyn_parallel(parallel_load_func, [(self.h5_file, group) for group in groups], desc='Loading HDF5 data')
        return merge_dicts(hdf5_data)

    def groups(self):
        with h5py.File(self.h5_file, 'r') as f:
            return list(f.keys())
