"""模型评估模块

该模块负责在测试集上评估训练好的模型性能，计算准确率、精确率、召回率和F1分数。
"""
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import model
from configuration import config
from preprocess.dataset import get_dataloader, DatasetType
from runner.predict import predict_batch
from model.bert_classifier import ProductClassifier


def evaluate_model(model, dataloader, device):
    """评估模型性能

    在给定的数据加载器上运行模型推理，收集所有预测结果和真实标签，
    计算并返回准确率、精确率、召回率和F1分数。

    Args:
        model: 待评估的模型实例
        dataloader: 测试数据加载器
        device: 计算设备（CPU或GPU）

    Returns:
        dict: 包含accuracy、precision、recall和f1-score的评估结果字典
    """
    all_labels = []
    all_predicts = []

    # 遍历测试数据集，收集真实标签和预测结果
    for batch in tqdm(dataloader, desc='评估'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].tolist()
        predict_result = predict_batch(input_ids, attention_mask, model, device)

        all_labels.extend(label)
        all_predicts.extend(predict_result)

    # 使用sklearn计算各项评估指标，精确率、召回率和F1使用macro平均
    accuracy = accuracy_score(all_labels, all_predicts)
    precision = precision_score(all_labels, all_predicts, average='macro')
    recall = recall_score(all_labels, all_predicts, average='macro')
    f1 = f1_score(all_labels, all_predicts, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }


def run_evaluate():
    """执行模型评估流程

    加载训练好的最佳模型，在测试集上进行评估，并打印各项指标结果。
    """
    # 选择计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型并加载最佳权重
    model = ProductClassifier().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))

    # 获取测试数据集加载器
    dataloader = get_dataloader(DatasetType.TEST)
    # 执行评估
    result = evaluate_model(model, dataloader, device)

    # 打印评估结果
    print('========== 评估结果 ==========')
    print(f"accuracy准确率:{result['accuracy']:.4f}")
    print(f"precision精确率:{result['precision']:.4f}")
    print(f"recall召回率:{result['recall']:.4f}")
    print(f"f1-score:{result['f1-score']:.4f}")
