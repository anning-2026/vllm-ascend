# Hunyuan-A13B-Instruct

## Run vllm-ascend on Multi-NPU with HunYuanMoEV1ForCausalLM

Introduction

Hunyuan-A13B-Instruct is a fine-grained hybrid expert model (MoE) developed by Tencent. This model has a total of 80 billion parameters, 13 billion activation parameters, supports 256K ultra-long contexts, and possesses native thought chain (CoT) reasoning capabilities.

This document provides detailed steps for deploying the model on multiple NPUs (Atlas 800 A2, 64G × 4) using a Conda virtual environment within the Modelers platform environment.

1.Environment Preparation

```bash
# Create and activate the Python 3.11 environment
conda create -n vllm python=3.11 -y
conda activate vllm

# Install and verify the component version.
pip install torch-npu==2.8.0
pip install vllm==0.13.0
pip install vllm-ascend==0.13.0rc1
```
2. Software stack version verification

The environment is based on CANN 8.3.RC2 built into the Modelers (模力方舟) platform, and successfully runs torch-npu 2.8.0, vLLM 0.13.0, and vLLM-Ascend 0.13.0rc1 through the Python 3.11.13 Conda environment.


## Deployment

Single-node Deployment (4-NPU)


```bash
# Activate the Conda environment
conda activate vllm

# Start the vLLM service
vllm serve /home/openmind/Hunyuan-A13B-Instruct \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 9999 \
    --served-model-name Hunyuan \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```
Key performance indicators (based on actual test logs):

Weighted memory usage: Each NPU has a static memory usage of approximately 37.46 GB.

Graph Compilation (ACL Graph): With PIECEWISE mode enabled, the system automatically captures the graph in approximately 26 seconds, which can significantly accelerate subsequent inference.

KV Cache Capacity: The remaining video memory can provide concurrent cache space for approximately 503,000 tokens.

## Functional Verification
```bash
curl http://localhost:9999/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/openmind/Hunyuan-A13B-Instruct",
        "messages": [{"role": "user", "content": "你好，请自我介绍一下。"}],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

Expected output:
```bash
<think>\n好的，用户让我自我介绍一下。首先，我需要明确自己的身份是腾讯开发的AI助手，名字是腾讯元宝，英文名Tencent Yuanbao。用户可能刚接触我，所以需要简明扼要地介绍核心功能。\n\n接下来，用户可能想知道我能做什么。我需要涵盖支持多模型切换，特别是Hunyuan-T1，这点很重要，因为模型能力直接影响用户体验。然后要提到多模态输入，比如文字、图片、文件，"
```
## Accuracy Evaluation
On the Modelers platform, the model was tested and verified using the AISBench tool on the GSM8K benchmark set: Under the e3c4be version configuration, the model achieved a high Ascend benchmark score of 93.63 in the accuracy generation mode.
