import torch
import numpy as np
import torch.utils.data
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from typing import Tuple
from utils.utils import get_adjacency_matrix
import pandas as pd
import os

def generate_sample_by_sliding_window(data, sample_len, step=1):

    sample = []

    for i in range(0, data.shape[0] - sample_len, step):

        sample.append(torch.unsqueeze(data[i:i+sample_len] , 0))
    
    if (data.shape[0] - sample_len) % step !=0 :
        sample.append(torch.unsqueeze(data[-sample_len:] , 0))

    sample = torch.concat(sample,dim=0)

    return sample

def generate_sample_index_by_sliding_window(data, sample_len, step=1):

    sample = []

    for i in range(0, data.shape[0] - sample_len, step):

        sample.append([i,i+sample_len])
    
    if (data.shape[0] - sample_len) % step !=0 :
        sample.append([data.shape[0]-sample_len,data.shape[0]])

    sample = torch.tensor(sample)

    return sample


class BasicDataset(torch.utils.data.Dataset):

    history  : torch.Tensor
    target   : torch.Tensor

    def __init__(self, history, target, sample_index,
                    sample_len, output_len, input_dim, output_dim, training=False) -> None:
        
        self.history = history
        self.target = target
        self.training = training

        self.sample_index = sample_index
        self.sample_len = sample_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):

        return self.sample_index.shape[0]

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Ttarget N Fin"],
    ]:

        l,r = self.sample_index[index]

        return (
            self.history[l:r-self.output_len, :, :self.input_dim],
            self.target[l+self.sample_len:r, :, :self.input_dim],
        )


class LastWindowDataset(BasicDataset):
    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Ttarget N Fin"],
    ]:

        l,r = self.sample_index[index]

        return (
            self.history[l:r, :, :self.input_dim],
            self.target[r-self.output_len:r, :, :self.input_dim],
        )


class MaskedDataset(BasicDataset):
    mask: torch.Tensor

    def __init__(self, history, target, mask, sample_index,
                    sample_len, output_len, input_dim, output_dim, training=False) -> None:
        super().__init__(
            history=history,
            target=target,
            sample_index=sample_index,
            sample_len=sample_len,
            output_len=output_len,
            input_dim=input_dim,
            output_dim=output_dim,
            training=training,
        )
        self.mask = mask

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Ttarget N Fin"],
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Tin N Fin"],
    ]:

        l,r = self.sample_index[index]
        history, target = super().__getitem__(index)
        context_mask = self.mask[l:r-self.output_len, :, :self.input_dim]

        return (
            history,
            target,
            context_mask,
            context_mask,  # eval_mask same as context_mask for forecast
        )


class LastWindowMaskedDataset(LastWindowDataset):
    mask: torch.Tensor

    def __init__(self, history, target, mask, sample_index,
                    sample_len, output_len, input_dim, output_dim, training=False) -> None:
        super().__init__(
            history=history,
            target=target,
            sample_index=sample_index,
            sample_len=sample_len,
            output_len=output_len,
            input_dim=input_dim,
            output_dim=output_dim,
            training=training,
        )
        self.mask = mask

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Ttarget N Fin"],
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Ttarget N Fin"],
    ]:

        l,r = self.sample_index[index]
        history, target = super().__getitem__(index)
        original_mask = self.mask[l:r, :, :self.input_dim].clone()
        context_mask = original_mask.clone()
        context_mask[-self.output_len:] = 0
        # eval_mask: original mask for target positions (used for loss/metric masking)
        eval_mask = original_mask[-self.output_len:, :, :]

        return (
            history,
            target,
            context_mask,
            eval_mask,
        )


class FullWindowMaskedDataset(BasicDataset):
    """Dataset for impute_full target_mode: reconstruct the entire window.

    History = full window (sample_len steps).
    Target = full window (sample_len steps).
    Mask = original missing pattern, NOT modified (no forced tail masking).
    Loss is computed only on mask=0 positions (truly missing data).
    """
    mask: torch.Tensor

    def __init__(self, history, target, mask, sample_index,
                    sample_len, output_len, input_dim, output_dim, training=False) -> None:
        # output_len == sample_len in impute_full mode
        super().__init__(
            history=history,
            target=target,
            sample_index=sample_index,
            sample_len=sample_len,
            output_len=output_len,
            input_dim=input_dim,
            output_dim=output_dim,
            training=training,
        )
        self.mask = mask

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Ttarget N Fin"],
        Float[Tensor, "Tin N Fin"],
        Float[Tensor, "Tin N Fin"],
    ]:

        l, r = self.sample_index[index]
        # History: full window (sample_len steps)
        history = self.history[l:r, :, :self.input_dim]
        # Target: full window (same as history — reconstruct all positions)
        target = self.target[l:r, :, :self.input_dim]
        # Mask: original pattern, no modification
        context_mask = self.mask[l:r, :, :self.input_dim].clone()

        return (
            history,
            target,
            context_mask,
            context_mask,  # eval_mask = context_mask (no forced tail zeros)
        )


class DataProvider():

    node_num : int
    features : int
    data  : torch.Tensor
    timestamp: torch.Tensor
    mask: torch.Tensor | None

    def __init__(self, data_path, adj_path ,dataset, node_shuffle_seed=None) -> None:

        self.dataset = dataset

        result = self.read_data(data_path, adj_path)
        self.data, self.node_num, self.features, \
        self.adj_mx, self.distance_mx, \
        self.timestamp = result[:6]
        self.mask = result[6] if len(result) == 7 else None

        if node_shuffle_seed is not None:
            rdm = np.random.RandomState(node_shuffle_seed)
            idx = np.arange(self.node_num)
            rdm.shuffle(idx)
            idx = torch.from_numpy(idx)
            self.data = self.data[:,idx,:]
            if self.mask is not None:
                self.mask = self.mask[:,idx,:]
            self.adj_mx = self.adj_mx[idx,:][:,idx]

    def read_data(self, data_path, adj_path=None):
        raise NotImplementedError

    def _build_dataset(self, data_range, sample_len, output_len, window_size,
                       input_dim, output_dim, training=True, target_mode="forecast"):
        data = self.data[data_range[0]:data_range[1]]
        sample_index = generate_sample_index_by_sliding_window(data, sample_len=window_size)
        if target_mode == "impute_full":
            dataset_cls = BasicDataset
            masked_dataset_cls = FullWindowMaskedDataset
        elif target_mode == "impute_last":
            dataset_cls = LastWindowDataset
            masked_dataset_cls = LastWindowMaskedDataset
        else:
            dataset_cls = BasicDataset
            masked_dataset_cls = MaskedDataset
        if self.mask is None:
            return dataset_cls(
                history=data,
                target=data,
                sample_index=sample_index,
                sample_len=sample_len,
                output_len=output_len,
                input_dim=input_dim,
                output_dim=output_dim,
                training=training,
            )

        mask = self.mask[data_range[0]:data_range[1]]
        return masked_dataset_cls(
            history=data,
            target=data,
            mask=mask,
            sample_index=sample_index,
            sample_len=sample_len,
            output_len=output_len,
            input_dim=input_dim,
            output_dim=output_dim,
            training=training,
        )

    def getdataset(self, sample_len, output_len, window_size, \
                    input_dim , output_dim , \
                   train_ratio, val_ratio, target_strategy, few_shot = 1,
                   target_mode = "forecast"):
        self.data = self.data.float()
        self.timestamp = self.timestamp.long()
        if self.mask is not None:
            self.mask = self.mask.float()

        all_len = self.data.shape[0]
        train_len = int(all_len * train_ratio)
        val_len = int(all_len * val_ratio)

        train_range = [0,int(train_len * few_shot)]
        val_range = [train_len, train_len+val_len]
        test_range = [train_len+val_len, all_len]

        train_dataset = self._build_dataset(
            train_range, sample_len, output_len, window_size, input_dim, output_dim,
            training=True, target_mode=target_mode
        )
        val_dataset = self._build_dataset(
            [val_range[0]-sample_len, val_range[1]],
            sample_len,
            output_len,
            window_size,
            input_dim,
            output_dim,
            training=True,
            target_mode=target_mode,
        )
        test_dataset = self._build_dataset(
            [test_range[0]-sample_len, test_range[1]],
            sample_len,
            output_len,
            window_size,
            input_dim,
            output_dim,
            training=True,
            target_mode=target_mode,
        )

        return train_dataset, val_dataset, test_dataset
                

    def getadj(self):

        return self.adj_mx, self.distance_mx


def generatetimestamp(start, periods, freq):

    time = pd.date_range(start=start,periods=periods,freq=freq)

    month = np.reshape(time.month, (-1, 1))
    dayofmonth = np.reshape(time.day, (-1, 1))
    dayofweek = np.reshape(time.weekday, (-1, 1))
    hour = np.reshape(time.hour, (-1, 1))
    minute = np.reshape(time.minute, (-1, 1))

    timestamp = np.concatenate((month, dayofmonth, dayofweek, hour, minute), -1)

    timestamp = torch.tensor(timestamp)

    return timestamp

timestampfun = {
    'PEMS08': lambda T : generatetimestamp(start='20160701 00:00:00',periods=T,freq='5min'),
    'PEMS07': lambda T : generatetimestamp(start='20170501 00:00:00',periods=T,freq='5min'),
    'PEMS04': lambda T : generatetimestamp(start='20180101 00:00:00',periods=T,freq='5min'),
    'PEMS03': lambda T : generatetimestamp(start='20180901 00:00:00',periods=T,freq='5min'),
    'NYCTAXI': lambda T : generatetimestamp(start='20160401 00:00:00',periods=T,freq='30min'),
    'CHIBIKE': lambda T : generatetimestamp(start='20160401 00:00:00',periods=T,freq='30min'),
    'XTRAFFIC': lambda T : generatetimestamp(start='20220101 00:00:00',periods=T,freq='5min'),
}

class PEMSFLOWProvider(DataProvider):

    def read_data(self, data_path,  adj_path = None ):

        data = torch.from_numpy(np.load(data_path)['data'][...,:])
        
        T, node_num, features = data.shape
        if 'PEMS03' in self.dataset:
            if adj_path is None:
                raise ValueError("adj_path is required for PEMS03")
            id_filename = adj_path.replace('csv','txt')
        else :
            id_filename = None
        if adj_path is None:
            raise ValueError(f"adj_path is required for {self.dataset}")
        adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num, id_filename)
        adj_mx = np.where(np.eye(node_num).astype('bool'),1,adj_mx)

        timestamp = timestampfun[self.dataset[:6]](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp

class PEMSMISSINGProvider(DataProvider):

    def read_data(self, data_path,  adj_path = None ):

        dir_name = os.path.dirname(data_path)
        fileName = os.path.basename(data_path)

        true_datapath = os.path.join(dir_name,fileName.replace('miss','true')) 
        miss_datapath = os.path.join(dir_name,fileName.replace('true','miss')) 

        miss_data = np.load(miss_datapath)
        mask = torch.from_numpy(miss_data['mask'][:, :, :].astype('long'))
        data = np.load(true_datapath)['data'].astype(np.float32)[:, :, :]
        data[np.isnan(data)] = 0
        data = torch.from_numpy(data)

        T, node_num, features = data.shape
 
        if adj_path is None:
            raise ValueError(f"adj_path is required for {self.dataset}")
        adj_mx, distance_mx = get_adjacency_matrix(adj_path, node_num)
        adj_mx = np.where(np.eye(node_num).astype('bool'),1,adj_mx)

        timestamp = timestampfun[self.dataset[:6]](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp,mask
    

class NYCTAXIProvider(DataProvider):

    def read_data(self, data_path,  adj_path = None ):

        data = torch.from_numpy(np.load(data_path)['data'][...,:])
        data = data.permute(1, 0, 2)
        
        T, node_num, features = data.shape

        adj_mx, distance_mx = np.ones((node_num,node_num)).astype(np.float32),np.ones((node_num,node_num)).astype(np.float32)
        timestamp = timestampfun[self.dataset](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp


class XTrafficProvider(DataProvider):
    """Provider for the Xtraffic California highway dataset.

    Expected data_path layout (directory mode)::

        data_path/
          p01_done.npy  ...  p12_done.npy   (T, N, F) float64, 5-min resolution
          adj_matrix.npy                  (N, N) float64 binary adjacency
          dis_matrix.npy                  (N, N) float64 distance matrix

    Alternatively a pre-concatenated ``.npz`` with key ``data``.
    NaN values are replaced with zero and an observation mask is produced.
    """

    def read_data(self, data_path, adj_path=None):
        import glob as _glob

        # --- load raw data --------------------------------------------------
        if os.path.isfile(data_path) and data_path.endswith('.npz'):
            raw = np.load(data_path)['data'].astype(np.float32)
        elif os.path.isdir(data_path):
            candidates = sorted(_glob.glob(os.path.join(data_path, 'p[0-9][0-9]_done.npy')))
            if not candidates:
                raise FileNotFoundError(f"No p*_done.npy partition files found in {data_path}")
            parts = [np.load(p).astype(np.float32) for p in candidates]
            raw = np.concatenate(parts, axis=0)
        else:
            raise ValueError(f"data_path must be a directory of p*_done.npy files or a .npz file: {data_path}")

        # Detect and fix node-first layout (year sub-dirs store (N,T,F) instead of (T,N,F))
        if raw.ndim == 3 and raw.shape[0] > raw.shape[2] and raw.shape[1] > 10000:
            if raw.shape[1] > raw.shape[0]:
                raw = raw.transpose(1, 0, 2)

        # --- mask from NaN positions ---------------------------------------
        mask = (~np.isnan(raw)).astype(np.float32)
        raw = np.nan_to_num(raw, nan=0.0, copy=False)

        data = torch.from_numpy(raw)
        mask_tensor = torch.from_numpy(mask)

        T, node_num, features = data.shape

        # --- adjacency / distance -------------------------------------------
        parent_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
        if adj_path is None:
            adj_path = os.path.join(parent_dir, 'adj_matrix.npy')
        if not os.path.exists(adj_path):
            raise FileNotFoundError(f"adjacency matrix not found: {adj_path}")
        adj_mx = np.load(adj_path).astype(np.float32)

        dis_path = os.path.join(os.path.dirname(adj_path), 'dis_matrix.npy')
        if os.path.exists(dis_path):
            distance_mx = np.load(dis_path).astype(np.float32)
        else:
            distance_mx = adj_mx.copy()

        adj_mx = np.where(np.eye(node_num).astype('bool'), 1, adj_mx)

        # --- timestamps -----------------------------------------------------
        timestamp = timestampfun[self.dataset](T)

        return data, node_num, features, \
               adj_mx, distance_mx, \
               timestamp, mask_tensor
