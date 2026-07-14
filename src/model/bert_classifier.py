"""商品分类BERT模型模块

该模块定义基于BERT预训练模型的商品分类器，采用BERT+线性层的架构，
通过冻结或微调BERT参数来实现商品标题的分类任务。
"""
from torch import nn
from transformers import AutoModel

from configuration import config


class ProductClassifier(nn.Module):
    """基于BERT的商品分类模型

    该模型由BERT预训练模型和一个线性分类层组成，使用BERT的[CLS] token输出
    作为句子表示，经过线性层映射到类别空间进行分类。

    Attributes:
        bert: BERT预训练模型
        linear: 线性分类层，将BERT的hidden_size映射到NUM_CLASSES
    """

    def __init__(self, freeze_bert=True):
        """初始化模型

        Args:
            freeze_bert: 是否冻结BERT的参数，默认True表示冻结
                         冻结时仅训练线性层，解冻时微调整个模型
        """
        super().__init__()
        # 加载BERT中文预训练模型
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
        # 根据freeze_bert参数决定是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = not freeze_bert
        # 定义线性分类层，输入维度为BERT的hidden_size，输出维度为类别数
        self.linear = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        """前向传播

        Args:
            input_ids: 输入序列的token索引张量，形状为[batch_size, seq_len]
            attention_mask: 注意力掩码张量，形状为[batch_size, seq_len]

        Returns:
            torch.Tensor: 分类结果logits，形状为[batch_size, NUM_CLASSES]

        各层输出形状说明：
            last_hidden: [batch_size, seq_len, hidden_size]
            cls_output: [batch_size, hidden_size]
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        # 获取[CLS] token的输出作为句子表示
        cls_output = last_hidden[:, 0, :]
        # 通过线性层得到分类结果
        return self.linear(cls_output)
