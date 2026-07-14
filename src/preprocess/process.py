"""原始数据预处理模块

该模块负责将原始文本数据转换为BERT模型可接受的格式，包括：
1. 读取原始数据文件
2. 过滤无效数据
3. 使用Tokenizer进行文本编码
4. 构建类别映射并转换标签
5. 保存处理后的数据集
"""
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

from configuration import config


def process_data():
    """执行数据预处理流程

    将原始的train.txt、test.txt、valid.txt文件转换为预处理后的数据集格式，
    包括文本编码和标签映射，最终保存到processed目录供后续训练和评估使用。
    """
    print("开始处理数据....")

    # 读取原始数据文件，使用Tab作为分隔符
    dataset_dict = load_dataset('csv', data_files={
        'train': str(config.RAW_DATA_DIR / 'train.txt'),
        'test': str(config.RAW_DATA_DIR / 'test.txt'),
        'valid': str(config.RAW_DATA_DIR / 'valid.txt')
    }, delimiter='\t')

    # 过滤掉label或text_a为空的数据，确保数据质量
    dataset_dict = dataset_dict.filter(lambda x: x['label'] is not None and x['text_a'] is not None)

    # 加载BERT中文预训练模型的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRE_TRAINED_DIR / 'bert-base-chinese'))

    # 定义文本编码函数，对每个样本进行分词、截断和padding
    def tokenize(batch):
        tokenized = tokenizer(batch['text_a'], truncation=True, padding='max_length', max_length=config.SEQ_LEN)
        return {'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
                }

    # 对所有数据集进行批量编码，并移除原始文本列
    dataset_dict = dataset_dict.map(tokenize, batched=True, remove_columns=['text_a'])

    # 从训练集中提取所有类别标签，构建类别到整数的映射
    all_labels = dataset_dict['train'].unique('label')
    print('\n类别总数:', len(all_labels))
    label2id = {label: i for i, label in enumerate(all_labels)}

    # 定义标签映射函数，将字符串标签转换为整数索引
    def map_label(batch):
        return {'label': [label2id[l] for l in batch['label']]}

    # 对所有数据集进行批量标签映射
    dataset_dict = dataset_dict.map(map_label, batched=True)

    # 创建ClassLabel对象，用于标签的整数与字符串互转
    class_label = ClassLabel(names=all_labels)
    # 将label列转换为ClassLabel类型
    dataset_dict = dataset_dict.cast_column('label', class_label)

    # 将处理后的数据集保存到磁盘
    dataset_dict['train'].save_to_disk(config.PROCESSED_DATA_DIR / 'train')
    dataset_dict['test'].save_to_disk(config.PROCESSED_DATA_DIR / 'test')
    dataset_dict['valid'].save_to_disk(config.PROCESSED_DATA_DIR / 'valid')

    print("处理数据完成!!!")
