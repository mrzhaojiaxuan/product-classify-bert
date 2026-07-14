"""Web服务请求和响应的数据模型定义模块

该模块使用Pydantic定义API接口的数据结构，确保请求参数校验和响应格式规范。
"""
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """商品分类预测请求模型

    Attributes:
        title: 待分类的商品标题文本
    """
    title: str


class PredictResponse(BaseModel):
    """商品分类预测响应模型

    Attributes:
        title: 原始输入的商品标题
        label: 模型预测的商品类别名称
    """
    title: str
    label: str
