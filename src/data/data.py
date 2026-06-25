import torch.utils.data
from data.dataprovider import PEMSFLOWProvider, PEMSMISSINGProvider, NYCTAXIProvider, XTrafficProvider

data_dict = {
    "PEMS08FLOW": PEMSFLOWProvider,
    "PEMS04FLOW": PEMSFLOWProvider,
    "PEMS03FLOW": PEMSFLOWProvider,
    "PEMS07FLOW": PEMSFLOWProvider,
    "PEMS08MISSING": PEMSMISSINGProvider,
    "PEMS04MISSING": PEMSMISSINGProvider,
    "PEMS03MISSING": PEMSMISSINGProvider,
    "PEMS07MISSING": PEMSMISSINGProvider,
    "NYCTAXI": NYCTAXIProvider,
    "CHITAXI": NYCTAXIProvider,
    "XTRAFFIC": XTrafficProvider,
}


def data_loader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0, prefetch_factor=2):
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        **kwargs,
    )


def load_data(
    dataset,
    sample_len,
    output_len,
    window_size,
    input_dim,
    output_dim,
    train_ratio,
    val_ratio,
    data_path,
    adj_path,
    target_strategy,
    batch_size,
    few_shot=1,
    node_shuffle_seed=None,
    num_workers=0,
    prefetch_factor=2,
    **kwargs,
):
    dataprovider = data_dict[dataset](data_path, adj_path, dataset, node_shuffle_seed)

    train_set, val_set, test_set = dataprovider.getdataset(
        sample_len=sample_len,
        output_len=output_len,
        window_size=window_size,
        input_dim=input_dim,
        output_dim=output_dim,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_strategy=target_strategy,
        few_shot=few_shot,
        target_mode=kwargs.get("target_mode", "forecast"),
    )

    train_loader = data_loader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = data_loader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = data_loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    node_num, features = dataprovider.node_num, dataprovider.features
    adj_mx, distance_mx = dataprovider.getadj()

    return train_loader, val_loader, test_loader, node_num, features, adj_mx, distance_mx
