"""项目配置模块

该模块定义项目的路径配置和超参数配置，所有模块共享这些配置项。
"""
from pathlib import Path

# 项目根目录，基于当前文件位置向上三级推导
ROOT_DIR = Path(__file__).parent.parent.parent

# 数据目录配置
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'           # 原始数据目录
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'  # 预处理后数据目录

# 输出目录配置
LOGS_DIR = ROOT_DIR / 'logs'           # TensorBoard日志目录
MODELS_DIR = ROOT_DIR / 'models'       # 模型保存目录
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'  # 预训练模型目录


# 模型超参数配置
SEQ_LEN = 64              # 输入序列最大长度
NUM_CLASSES = 30          # 类别总数

# 训练超参数配置
BATCH_SIZE = 32           # 批量大小
LEARNING_RATE = 1e-5      # 学习率
EPOCHS = 10               # 训练轮数
