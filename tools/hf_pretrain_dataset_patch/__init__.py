from .pretrain_dataset import HFPretrainDataset
from .utils import get_batch_on_this_tp_rank_original

def build_pretrain_dataset_from_original(cache_path):
    train_dataset = HFPretrainDataset(cache_path)
    return train_dataset, train_dataset, train_dataset