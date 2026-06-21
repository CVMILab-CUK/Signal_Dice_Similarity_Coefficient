import os
import random
import numpy as np
import torch
import torch.multiprocessing as _mp
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


def _seed_worker(worker_id):
    """Seed each DataLoader worker AND switch the worker's tensor sharing strategy
    to 'file_system' so it doesn't write into the 64MB container /dev/shm. With
    TMPDIR=/workspace/tmp the shared-storage files land on the 716GB /workspace."""
    # Force every worker into file_system sharing strategy. Set BEFORE any tensor
    # touches shared memory in this worker.
    os.environ.setdefault("TMPDIR", "/workspace/tmp")
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    try:
        _mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_generator(seed):
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _safe_num_workers(requested):
    """Cap num_workers to limit /dev/shm pressure in container with 64MB shm.
    Each ECL batch (~2MB) x prefetch_factor=1 x N workers must fit < 64MB.
    Cap at 2 workers globally; if caller passed 0, respect that."""
    if requested is None or requested == 0:
        return 0
    return min(int(requested), 2)


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    seed = getattr(args, 'seed', None)

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=_safe_num_workers(args.num_workers),
            prefetch_factor=1 if _safe_num_workers(args.num_workers) > 0 else None,
            persistent_workers=_safe_num_workers(args.num_workers) > 0,
            drop_last=drop_last,
            worker_init_fn=_seed_worker,
            generator=_make_generator(seed))
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            worker_init_fn=_seed_worker,
            generator=_make_generator(seed)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=_safe_num_workers(args.num_workers),
            prefetch_factor=1 if _safe_num_workers(args.num_workers) > 0 else None,
            persistent_workers=_safe_num_workers(args.num_workers) > 0,
            drop_last=drop_last,
            worker_init_fn=_seed_worker,
            generator=_make_generator(seed))

        print(flag, len(data_set), len(data_loader))
        return data_set, data_loader
