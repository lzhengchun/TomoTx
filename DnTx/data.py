import numpy as np 
import h5py, logging
from torch.utils.data import Dataset

class SinogramDataset(Dataset):
    def __init__(self, ifn, params, world_size=None, rank=None):
        ds_cfg = params['dataset']

        h5fd = h5py.File(ifn, 'r')
        n_samples = h5fd[ds_cfg['dkey']].shape[0]

        if world_size is None or rank is None:
            si, ei = 0, None
        else:
            npr = n_samples // world_size
            si = rank * npr
            ei = (rank + 1) * npr
        logging.info(f"rank {rank} loading data from index {si} to {ei}")
        self.targets  = h5fd[ds_cfg['dkey']][si:ei]
        h5fd.close()

        self.features = np.random.poisson(ds_cfg['ldfactor'] * self.targets) 
        # norm or not will influence pos-encodding
        if ds_cfg['inorm']['norm']:
            if ds_cfg['inorm'].get('mean4norm') is None:
                _avg = self.features.mean()
                _std = self.features.std()
                self.features = ((self.features - _avg) / _std).astype(np.float32)
                logging.info(f'rank {rank} features are normalized with mean: {_avg}, std: {_std}')
            else:
                self.features = ((self.features - ds_cfg['inorm']['mean4norm']) / ds_cfg['inorm']['std4norm']).astype(np.float32)

        if ds_cfg['onorm']['norm']:
            if ds_cfg['onorm'].get('mean4norm') is None:
                _avg = self.targets.mean(axis=(1,2))
                _std = self.targets.std( axis=(1,2))
                self.targets = ((self.targets - _avg) / _std).astype(np.float32)
                logging.info(f'rank {rank} targets are normalized with mean: {_avg}, std: {_std}')
            else:
                self.targets = ((self.targets - ds_cfg['onorm']['mean4norm']) / ds_cfg['onorm']['std4norm']).astype(np.float32)

        self.len    = self.targets.shape[0]
        self.seqlen = self.targets.shape[-2] # angle
        self.cdim   = self.targets.shape[-1] # resolution
        self.shape  = self.targets.shape

    def __getitem__(self, idx):
            return self.features[idx][None].astype(np.float32), self.targets[idx][None].astype(np.float32)

    def __len__(self):
        return self.len