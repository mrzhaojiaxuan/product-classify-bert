"""模型训练模块

该模块负责模型的训练流程，包括数据加载、模型初始化、训练循环、
早停策略和模型保存，支持断点续训和混合精度训练。
"""
import time

import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configuration import config
from model.bert_classifier import ProductClassifier
from preprocess.dataset import get_dataloader, DatasetType


class EarlyStopping:
    """早停策略类

    当验证损失不再下降时，提前终止训练以防止过拟合。
    连续patience个epoch损失没有改善时触发早停。

    Attributes:
        best_loss: 当前最佳损失值
        counter: 连续损失未改善的epoch计数
        patience: 容忍的最大连续未改善epoch数
    """

    def __init__(self, patience=3):
        """初始化早停策略

        Args:
            patience: 容忍的最大连续未改善epoch数，默认3
        """
        self.best_loss = float('inf')
        self.counter = 0
        self.patience = patience

    def should_stop(self, avg_loss, model, path):
        """判断是否应该停止训练

        如果当前损失优于最佳损失，则更新最佳损失并保存模型；
        否则增加计数器，当计数器达到patience时返回True表示应停止训练。

        Args:
            avg_loss: 当前epoch的平均损失
            model: 当前模型实例
            path: 最佳模型保存路径

        Returns:
            bool: 是否应该停止训练
        """
        if avg_loss < self.best_loss:
            self.counter = 0
            self.best_loss = avg_loss
            torch.save(model.state_dict(), path)
            print('保存最优模型')
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True


def run_one_epoch(model, dataloader, loss_function, device, scaler=None, optimizer=None, is_train=True):
    """运行单个epoch的训练或验证

    根据is_train参数决定是训练模式还是验证模式，
    遍历数据加载器执行前向传播和反向传播（仅训练模式），
    返回当前epoch的平均损失。

    Args:
        model: 待训练或验证的模型
        dataloader: 数据加载器
        loss_function: 损失函数
        device: 计算设备（CPU或GPU）
        scaler: 梯度缩放器，用于混合精度训练
        optimizer: 优化器，训练模式必需
        is_train: 是否为训练模式，默认True

    Returns:
        float: 当前epoch的平均损失
    """
    total_loss = 0
    # 设置模型模式：训练模式启用dropout等，验证模式禁用
    if is_train:
        model.train()
    else:
        model.eval()

    # 控制梯度计算：训练模式启用梯度计算，验证模式禁用
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc=('训练' if is_train else '验证')):
            # 将数据转移到指定设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            # 混合精度训练上下文
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                # 前向传播
                outputs = model(input_ids, attention_mask)
                # 计算损失
                loss = loss_function(outputs, label)

            # 反向传播和参数更新（仅训练模式）
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

    # 返回平均损失
    return total_loss / len(dataloader)


def train():
    """执行模型训练流程

    完整的训练流程包括：设备选择、数据加载、模型初始化、损失函数和优化器配置、
    TensorBoard日志记录、早停策略设置、断点续训检查、训练循环执行。
    """
    # 选择计算设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据和验证数据
    train_dataloader = get_dataloader()
    valid_dataloader = get_dataloader(DatasetType.VALID)

    # 初始化模型，冻结BERT参数设为False表示微调
    model = ProductClassifier(freeze_bert=False).to(device)

    # 定义损失函数：交叉熵损失
    loss_function = torch.nn.CrossEntropyLoss()

    # 定义优化器：Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 初始化TensorBoard日志记录器，日志目录包含时间戳
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    # 初始化早停策略
    earlystopping = EarlyStopping()

    # 初始化梯度缩放器，用于混合精度训练
    scaler = torch.amp.GradScaler(device.type)

    # 设置起始epoch，默认从1开始
    start_epoch = 1

    # 检查是否存在checkpoint文件，支持断点续训
    checkpoint_path = config.MODELS_DIR / 'checkpoint.pt'
    if checkpoint_path.exists():
        print('已找到checkpoint文件，继续训练')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        earlystopping.best_loss = checkpoint['best_loss']
        earlystopping.counter = checkpoint['counter']
        start_epoch = checkpoint['epoch']
    else:
        print('未找到checkpoint文件，从新开始训练')

    # 训练主循环
    best_loss = float('inf')
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f"========== EPOCH {epoch} ==========")

        # 训练一个epoch
        train_avg_loss = run_one_epoch(model, train_dataloader, loss_function, device, scaler, optimizer)
        # 验证一个epoch
        valid_avg_loss = run_one_epoch(model, valid_dataloader, loss_function, device, is_train=False)

        # 打印当前epoch的损失
        print(f"loss: {train_avg_loss:.4f}")
        print(f"valid_loss: {valid_avg_loss:.4f}")

        # 记录TensorBoard日志
        writer.add_scalar('Loss/Train', train_avg_loss, epoch)
        writer.add_scalar('Loss/Valid', valid_avg_loss, epoch)

        # 检查是否触发早停
        if earlystopping.should_stop(train_avg_loss, model, config.MODELS_DIR / 'best.pt'):
            print('触发早停策略,训练提前停止')
            break

        # 保存训练状态（checkpoint），用于断点续训
        checkpoint = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(),
                      "best_loss": earlystopping.best_loss,
                      "counter": earlystopping.counter,
                      "epoch": epoch}
        torch.save(checkpoint, config.MODELS_DIR / 'checkpoint.pt')

    # 关闭TensorBoard日志记录器
    writer.close()
