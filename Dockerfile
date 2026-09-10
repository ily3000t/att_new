# 指定基础镜像
FROM python:3.8-slim-buster

# 设置工作目录
WORKDIR /

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

