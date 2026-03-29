# Hunyuan-A13B-Instruct

## Introduction

Run vllm-ascend on Multi-NPU with HunYuanMoEV1ForCausalLM.

Hunyuan-A13B-Instruct is a fine-grained hybrid expert model (MoE) developed by Tencent. This model has a total of 80 billion parameters, 13 billion activation parameters, supports 256K ultra-long contexts, and possesses native thought chain (CoT) reasoning capabilities.

This document provides detailed steps for deploying the model on multiple NPUs (Atlas 800 A2, 64G × 4) using a Conda virtual environment within the Modelers platform environment.

## Environment Preparation

### Installation

Run docker container:

```{code-block} bash
   :substitutions:
# Update the vllm-ascend image
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Build from source:

```bash
# Install vLLM.
git clone --depth 1 --branch v0.17.0 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch v0.17.0rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e .
cd ..
```

### Software Stack Version Verification

The environment is based on CANN 8.5.1 built into the GiteeAI (模力方舟) platform, and successfully runs vLLM 0.17.0, and vLLM-Ascend 0.17.0rc1 through the Python 3.11.6 Conda environment.

## Deployment

### Single-node Deployment (4-NPU)

```bash
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/data
export MODEL_PATH="Hunyuan-A13B-Instruct"
# Start the vLLM service
vllm serve ${MODEL_PATH} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 9999 \
    --served-model-name Hunyuan \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --async-scheduling
```

### Key Performance Indicators

Based on verified CANN 8.5.1 test logs:

- Memory usage for weights: each NPU has a static memory usage of approximately 37.46 GB.
- Graph compilation (ACL Graph): with PIECEWISE mode enabled, the system automatically captures the graph in approximately 18 seconds, which can significantly accelerate subsequent inference.
- KV cache capacity: the remaining NPU memory can provide concurrent cache space for approximately 529,152 tokens.

## Functional Verification

```bash
curl http://localhost:9999/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hunyuan",
        "messages": [{"role": "user", "content": "Give me a short introduction to large language models."}],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

Expected output:

```json
{"id":"chatcmpl-9a60df2b23bb539f","object":"chat.completion","created":1774751760,"model":"Hunyuan","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\nOkay, I need to write a short introduction to large language models. Let me start by recalling what I know. First, what are LLMs? They're machine learning models trained on vast amounts of text data. The key here is \"large\"—so they have a huge number of parameters. Maybe mention the scale, like billions or trillions of parameters.\n\nThen, how are they trained? They're trained on diverse text sources—books, websites, articles, etc. The","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":12,"total_tokens":112,"completion_tokens":100,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

## Accuracy Evaluation

On the GiteeAI platform, the model was tested and verified using the AISBench tool on the GSM8K benchmark set: Under the 7cd45e version configuration, the model achieved an accuracy of 94.77% in the accuracy generation mode.

```bash
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --summarizer example --debug
```

Expected output:

```bash
03/29 03:20:03 - AISBench - INFO - Running 1-th replica of evaluation
03/29 03:20:03 - AISBench - INFO - Task [vllm-api-general-chat/gsm8k]: {'accuracy': 94.76876421531463}
03/29 03:20:03 - AISBench - INFO - time elapsed: 2.15s
03/29 03:20:04 - AISBench - INFO - Evaluation tasks completed.
03/29 03:20:04 - AISBench - INFO - Summarizing evaluation results...
dataset    version    metric    mode      vllm-api-general-chat
---------  ---------  --------  ------  -----------------------
gsm8k      7cd45e     accuracy  gen                       94.77
03/29 03:20:04 - AISBench - INFO - write summary to /data/outputs/default/20260329_025345/summary/summary_20260329_025345.txt
03/29 03:20:04 - AISBench - INFO - write csv to /data/outputs/default/20260329_025345/summary/summary_20260329_025345.csv
```

The markdown formatted result is as follows:

| dataset | version | metric | mode | vllm-api-general-chat |
| --- | --- | --- | --- | --- |
| gsm8k | 7cd45e | accuracy | gen | 94.77 |
