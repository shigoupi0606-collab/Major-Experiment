# 大模型知识编辑实验 (Knowledge Editing for LLMs)

**学生：** 侍婉婷
**实验方向：** 方向‑03：大模型知识编辑  
**日期：** 2026年5月19日

## 一、实验概述

本实验旨在利用知识编辑技术，高效修改大语言模型（LLM）中的过时或错误事实，避免重新训练的巨大开销。实验基于 **Qwen2.5‑0.5B** 模型，使用 **EasyEdit** 开源框架，实现了 **ROME**（单条编辑）和 **MEMIT**（批量编辑）两种主流算法，并评估了编辑成功率（ES）、泛化性（PS）和局部性（NS）三个核心指标。

## 二、实验环境

- **云服务器配置**：阿里云 ECS（8 vCPU, 32GB RAM, NVIDIA T4 16GB GPU）
- **操作系统**：Ubuntu 22.04 LTS
- **Python 版本**：3.10
- **CUDA 版本**：12.8（驱动兼容 12.4 运行时）

### 依赖安装

# 创建虚拟环境
conda create -n easyedit_env python=3.10 -y
conda activate easyedit_env

# 安装 PyTorch（CUDA 12.4 兼容 12.8 驱动）
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements.txt

# 安装 EasyEdit
pip install easyeditor