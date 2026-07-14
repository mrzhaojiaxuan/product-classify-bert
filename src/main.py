"""项目入口模块

该模块是项目的命令行入口，通过解析命令行参数执行不同的功能：
- train: 训练模型
- predict: 交互式预测
- evaluate: 评估模型
- preprocess: 数据预处理
- serve: 启动Web服务

使用方式：
    python main.py <command>

示例：
    python main.py train      # 开始训练
    python main.py predict    # 交互式预测
    python main.py evaluate   # 评估模型
    python main.py preprocess # 数据预处理
    python main.py serve      # 启动Web服务
"""
from argparse import ArgumentParser

if __name__ == '__main__':
    # 创建命令行参数解析器
    arg_parser = ArgumentParser(description='商品分类BERT模型 - 命令行工具')
    # 添加命令参数，限制可选值为5个功能命令
    arg_parser.add_argument('command', choices=['train', 'predict', 'evaluate', 'preprocess', 'serve'],
                            help='要执行的命令：train(训练)、predict(预测)、evaluate(评估)、preprocess(预处理)、serve(服务)')
    args = arg_parser.parse_args()

    # 根据命令参数执行对应功能
    command = args.command

    if command == 'train':
        from runner.train import train

        train()
    elif command == 'predict':
        from runner.predict import run_predict

        run_predict()
    elif command == 'evaluate':
        from runner.evaluate import run_evaluate

        run_evaluate()
    elif command == 'preprocess':
        from preprocess.process import process_data

        process_data()
    elif command == 'serve':
        from web.app import run_app

        run_app()
