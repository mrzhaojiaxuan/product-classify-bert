"""Web服务预测业务逻辑模块

该模块负责初始化预测所需的资源（模型、Tokenizer、类别映射），
并提供统一的商品标题分类预测接口供路由层调用。
"""
import torch
from transformers import AutoTokenizer

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataset, DatasetType
from runner.predict import predict

# 模块加载时初始化预测资源

# 选择计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化分类模型并加载最佳权重
model = ProductClassifier().to(device)
model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))

# 加载BERT中文预训练模型的Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

# 加载训练数据集以获取类别映射信息
dataset = get_dataset(DatasetType.TRAIN)
class_label = dataset.features['label']


def predict_title(text):
    """预测商品标题的类别

    封装预测逻辑，接收商品标题文本并返回预测的类别名称。

    Args:
        text: 待分类的商品标题文本

    Returns:
        str: 预测的商品类别名称
    """
    return predict(text, model, tokenizer, device, class_label)
