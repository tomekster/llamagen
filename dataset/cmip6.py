from torch.utils.data import Dataset
import torch
import h5netcdf
import os
import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler

def get_nc_vars(h5nc_file):
        vars = []
        batch_sizes = []
        for k in h5nc_file.variables.keys():
            dim = h5nc_file.variables[k].shape
            if len(dim) >= 3 and dim[-1] >= 64 and dim[-2] >= 128:
                vars.append(h5nc_file.variables[k])
                batch_sizes.append(np.prod(dim)/(dim[-1]*dim[-2]))
        return vars, batch_sizes

class CMIP6NCDataset(Dataset):
    def __init__(self, args, **kwargs):
        self.args = args
        self.rank = kwargs["rank"]
        self.world_size = kwargs["world_size"]
        # TODO: cycle back to beginning of dataset if not enough data
        self.global_file_list = [f.strip() for f in open(args.data_path, "r").readlines()]
        self.local_file_list = []
        self.local_var_list = []
        self.local_batch_sizes = []
        file_sizes = [os.path.getsize(file) for file in self.global_file_list]
        total_size = sum(file_sizes)
        total_size_per_rank = total_size / self.world_size
        size_accu = 0
        for i, f in enumerate(self.global_file_list):
            size_accu += file_sizes[i]
            if size_accu >= total_size_per_rank * self.rank and size_accu < total_size_per_rank * (self.rank + 1):
                h5nc_file = h5netcdf.File(f, "r")
                self.local_file_list.append(h5nc_file)
                vars, batch_sizes = get_nc_vars(h5nc_file)
                self.local_var_list += vars
                self.local_batch_sizes += batch_sizes
            if size_accu >= total_size_per_rank * (self.rank + 1):
                break
        self.local_batch_sizes_cumsum = np.cumsum(self.local_batch_sizes)
        self.length = sum(self.local_batch_sizes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise StopIteration()
        var_idx = np.searchsorted(self.local_batch_sizes_cumsum, idx)
        local_idx = self.local_batch_sizes_cumsum[var_idx] - idx
        var = self.local_var_list[var_idx]
        fillvalue = var.attrs['_FillValue']
        local_idx_tuple = np.unravel_index(local_idx, var.shape[:-2])
        tag = f"{var.name}_ilevel{local_idx_tuple[-1]}"
        data = var[local_idx_tuple]
        if np.any(data == fillvalue):
            # handle NaN value
            return self.__getitem__((idx + 1) % self.length)
        data = torch.from_numpy(data)
        return data, tag
    
def build_cmip6_nc(args, **kwargs):
    dataset = CMIP6NCDataset(args, **kwargs)
    if kwargs["shuffle"]:
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(kwargs["seed"]))
    else:
        sampler = SequentialSampler(dataset)
    return dataset, sampler