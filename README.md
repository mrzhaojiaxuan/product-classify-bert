# 商品标题分类系统

基于BERT预训练模型的商品标题分类系统，实现商品标题到30个类别的自动归类。采用PyTorch框架，使用`bert-base-chinese`作为特征提取器，结合线性分类层完成多分类任务。

## 功能特性

- **数据预处理**：支持原始数据读取、过滤、编码和标签映射
- **模型训练**：支持早停策略、断点续训和混合精度训练
- **模型评估**：计算准确率、精确率、召回率和F1分数
- **交互式预测**：命令行输入商品标题，实时返回分类结果
- **RESTful API**：基于FastAPI的Web服务，提供分类预测接口

## 项目结构

```
product_classify_bert/
├── data/
│   ├── raw/          # 原始数据
│   └── processed/    # 预处理后数据
├── logs/             # TensorBoard日志
├── models/           # 模型文件
├── pretrained/       # 预训练模型
└── src/
    ├── configuration/   # 配置模块
    │   └── config.py
    ├── model/           # 模型模块
    │   └── bert_classifier.py
    ├── preprocess/      # 数据预处理模块
    │   ├── dataset.py
    │   └── process.py
    ├── runner/          # 运行器模块
    │   ├── train.py
    │   ├── predict.py
    │   └── evaluate.py
    ├── web/             # Web服务模块
    │   ├── app.py
    │   ├── predict_router.py
    │   ├── service.py
    │   └── schemas.py
    └── main.py          # 命令行入口
```

## 环境依赖

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- Datasets
- scikit-learn
- tqdm
- TensorBoard

## 安装步骤

```bash
# 安装依赖
pip install torch transformers fastapi uvicorn datasets scikit-learn tqdm tensorboard

# 下载预训练模型（首次运行时自动下载，或手动下载到 pretrained/bert-base-chinese/）
# 模型：bert-base-chinese，可从 Hugging Face 下载
```

## 使用方法

### 数据预处理

```bash
python src/main.py preprocess
```

### 模型训练

```bash
python src/main.py train
```

### 模型评估

```bash
python src/main.py evaluate
```

### 交互式预测

```bash
python src/main.py predict
```

### 启动Web服务

```bash
python src/main.py serve
```

服务启动后访问 `http://localhost:8000/predict` 进行预测。

## API接口

### POST /predict

请求体：

```json
{
    "title": "商品标题文本"
}
```

响应体：

```json
{
    "title": "商品标题文本",
    "label": "预测类别"
}
```

## 训练参数

在 `src/configuration/config.py` 中可配置：

- `SEQ_LEN`: 输入序列长度 (默认: 64)
- `NUM_CLASSES`: 类别总数 (默认: 30)
- `BATCH_SIZE`: 批量大小 (默认: 32)
- `LEARNING_RATE`: 学习率 (默认: 1e-5)
- `EPOCHS`: 训练轮数 (默认: 10)
