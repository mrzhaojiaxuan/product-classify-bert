"""FastAPI Web服务应用入口模块

该模块创建FastAPI应用实例并注册路由，提供商品分类预测的RESTful API服务。
"""
import uvicorn
from fastapi import FastAPI

from web.predict_router import predict_router

# 创建FastAPI应用实例
app = FastAPI(title="商品分类预测服务", description="基于BERT模型的商品标题分类API服务")

# 注册预测路由
app.include_router(predict_router)


def run_app():
    """启动Web服务

    使用uvicorn在本地8000端口启动FastAPI应用，提供HTTP服务。
    """
    uvicorn.run("web.app:app", port=8000)
