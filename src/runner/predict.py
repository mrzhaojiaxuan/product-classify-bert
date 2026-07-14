"""模型预测模块

该模块提供模型推理功能，支持批量预测和单条文本预测，
将预测结果从整数索引转换为类别名称。
"""
import torch
from transformers import AutoTokenizer

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataset, DatasetType


def predict_batch(input_ids, attention_mask, model, device):
    """批量预测函数

    在评估模式下运行模型推理，对一批输入数据进行分类预测，
    返回预测结果的整数索引列表。

    Args:
        input_ids: 输入序列的token索引张量，形状为[batch_size, seq_len]
        attention_mask: 注意力掩码张量，形状为[batch_size, seq_len]
        model: 预训练好的分类模型
        device: 计算设备（CPU或GPU）

    Returns:
        list: 预测类别的整数索引列表
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    predicts = torch.argmax(outputs, dim=1)
    return predicts.tolist()


def predict(text, model, tokenizer, device, class_label):
    """单条文本预测函数

    对单个商品标题进行分类预测，返回预测的类别名称。

    Args:
        text: 待分类的商品标题文本
        model: 预训练好的分类模型
        tokenizer: BERT分词器
        device: 计算设备（CPU或GPU）
        class_label: ClassLabel对象，用于整数索引到类别名称的转换

    Returns:
        str: 预测的商品类别名称
    """
    # 使用Tokenizer对文本进行编码，包括padding和截断
    encoded = tokenizer([text], padding="max_length", truncation=True, max_length=config.SEQ_LEN, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 调用批量预测函数，获取预测结果
    batch_result = predict_batch(input_ids, attention_mask, model, device)
    result = batch_result[0]

    # 将整数索引转换为类别名称并返回
    return class_label.int2str(result)


def run_predict():
    """交互式预测入口

    加载模型和Tokenizer，提供命令行交互式预测功能，
    用户输入商品标题后返回预测类别，输入q或quit退出。
    """
    # 选择计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型并加载最佳权重
    model = ProductClassifier().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))

    # 加载BERT中文预训练模型的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRE_TRAINED_DIR / 'bert-base-chinese'))

    # 加载训练数据集以获取类别映射信息
    dataset = get_dataset(DatasetType.TRAIN)
    class_label = dataset.features['label']

    # 进入交互式预测循环
    print("开始预测...")
    print("请输入商品标题")
    while True:
        text = input("> ")
        if text in ['q', 'quit']:
            break
        if not text:
            continue

        cls = predict(text, model, tokenizer, device, class_label)
        print('所属类别:', cls)
