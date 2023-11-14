import pytorch_lightning as pl
import torch


def seed_everything(seed=42, workers=True):
    '''
    seed everything and make the model deterministic to ensure the reproducibility
    :param seed: 随机种子
    :return: None
    '''
    pl.seed_everything(seed=seed, workers=workers)


def get_use_gpu_list(gpus):
    device_count = torch.cuda.device_count()
    if type(gpus) == int:
        if gpus == -1:
            gpu_list = [i for i in range(device_count)]
        else:
            gpu_list = [i for i in range(gpus)]
    elif type(gpus) == str:
        if ',' in gpus:
            gpu_list = gpus.split(',')
        else:
            if int(gpus) == -1:
                gpu_list = [i for i in range(device_count)]
            else:
                gpu_list = [i for i in range(int(gpus))]
    elif type(gpus) == list:
        gpu_list = gpus
    else:
        raise RuntimeError(f'Param gpus have illegal type: {gpus}')
    return gpu_list


def is_single_gpu(gpus):
    gpu_list = get_use_gpu_list(gpus)
    return len(gpu_list) == 1


if __name__ == '__main__':
    print('gpu_list:', get_use_gpu_list(0))
    print('gpu_list:', get_use_gpu_list(3))
    seed_everything(seed=42, workers=True)
    a = torch.rand([3, 5])
    print(a)
