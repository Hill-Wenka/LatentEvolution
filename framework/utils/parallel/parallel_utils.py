import multiprocessing as mp

from tqdm.notebook import tqdm


def asyn_parallel(parallel_func, params, kwds=None, cpu_num=mp.cpu_count(), desc=None):
    kwds = {} if kwds is None else kwds
    pbar = tqdm(total=len(params))
    desc = desc if desc is not None else 'Parallel Running, cpu_num: %d' % cpu_num
    pbar.set_description(desc)
    update = lambda *args: pbar.update()

    p = mp.Pool(cpu_num)
    results = [p.apply_async(parallel_func, args=param, kwds=kwds, callback=update) for param in params]
    results = [p.get() for p in results]
    p.close()
    p.join()
    return results


def syn_parallel(parallel_func, params, cpu_num=mp.cpu_count()):
    pbar = tqdm(total=len(params))
    pbar.set_description('Parallel Running')
    update = lambda *args: pbar.update()

    p = mp.Pool(cpu_num)
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_dict = manager.dict()
    results = [p.apply_async(parallel_func, args=(*param, managed_dict, managed_locker), callback=update) for param in
               params]
    [p.get() for p in results]
    managed_dict = managed_dict._getvalue()
    return managed_dict
