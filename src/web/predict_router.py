"""商品分类预测API路由模块

该模块定义商品分类预测的RESTful API端点，处理HTTP请求并返回预测结果。
"""
from fastapi import APIRouter, HTTPException

from web.schemas import PredictRequest, PredictResponse
from web.service import predict_title

# 创建API路由实例
predict_router = APIRouter()


@predict_router.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    """商品标题分类预测接口

    接收商品标题文本，调用预测服务返回分类结果。

    Args:
        req: 预测请求对象，包含商品标题

    Returns:
        PredictResponse: 预测响应对象，包含原始标题和预测类别

    Raises:
        HTTPException: 当标题为空时返回400错误
    """
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title不能为空")
    label = predict_title(title)
    return PredictResponse(title=title, label=label)
