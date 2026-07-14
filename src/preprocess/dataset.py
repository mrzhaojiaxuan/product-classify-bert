"""数据集加载与数据加载器构建模块

该模块提供从磁盘加载预处理后数据集的功能，并构建PyTorch DataLoader用于训练、验证和测试。
"""
from enum import Enum

from datasets import load_from_disk
from torch.utils.data import DataLoader
from configuration import config


class DatasetType(Enum):
    """数据集类型枚举

    定义训练集、验证集和测试集三种数据类型，用于标识不同用途的数据集。
    """
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


def get_dataset(data_type=DatasetType.TRAIN):
    """加载指定类型的数据集

    从预处理数据目录加载指定类型的数据集，返回datasets.Dataset对象。

    Args:
        data_type: 数据集类型，默认为训练集

    Returns:
        datasets.Dataset: 加载后的数据集对象
    """
    path = str(config.PROCESSED_DATA_DIR / data_type.value)
    return load_from_disk(path)


def get_dataloader(data_type=DatasetType.TRAIN):
    """构建指定类型的数据加载器

    加载数据集并转换为PyTorch张量格式，返回DataLoader对象用于批量加载数据。

    Args:
        data_type: 数据集类型，默认为训练集

    Returns:
        DataLoader: PyTorch数据加载器，包含input_ids、attention_mask和label字段
    """
    path = str(config.PROCESSED_DATA_DIR / data_type.value)
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
