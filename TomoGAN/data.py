import h5py, logging, random
from torch.utils.data import Dataset
import numpy as np 

class DnDataset(Dataset):
    def __init__(self, ih5, params):
        self.psz = params['train']['psz']
        ds_cfg = params['dataset']

        N = h5py.File(ih5, 'r')[ds_cfg['xkey']].shape[0]

        with h5py.File(ih5, 'r') as fp:
            self.features = fp[ds_cfg['xkey']][:]
            self.targets  = fp[ds_cfg['ykey']][:]

        if ds_cfg['norm']:
            if ds_cfg['mean4norm'] is None:
                _avg = self.features.mean()
                _std = self.features.std()
                self.features = ((self.features - _avg) / _std).astype(np.float32)
                logging.info(f'features from {ih5} are normalized with computed mean: {_avg}, std: {_std}')
            else:
                self.features = ((self.features - ds_cfg['mean4norm']) / ds_cfg['std4norm']).astype(np.float32)
                logging.info(f"features from {ih5} are normalized with provided mean: {ds_cfg['mean4norm']}, std: {ds_cfg['std4norm']}")

        self.dim = self.features.shape

    def __getitem__(self, idx):
        if self.features.shape[-2] == self.psz:
            rst, cst = 0, 0
        else:
            rst = np.random.randint(0, self.features.shape[-2]-self.psz)
            cst = np.random.randint(0, self.features.shape[-1]-self.psz)

        inp = self.features[idx, np.newaxis, rst:rst+self.psz, cst:cst+self.psz]
        out = self.targets [idx, np.newaxis, rst:rst+self.psz, cst:cst+self.psz]
        return inp, out

    def __len__(self):
        return self.features.shape[0]


