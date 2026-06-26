# Rethinking Reasoning for Next-Generation Agentic AI

Official repository for evaluating **reasoning efficiency in Agentic AI**. This framework benchmarks autonomous LLM agents by analyzing reasoning behavior, tool usage, latency, token consumption, and task performance across multiple foundation models.

**GitHub:** https://github.com/juealcs/Reasoning_Agentic_AI

---

## Features

- Agentic AI benchmark built with LangGraph
- Dynamic tool routing
- Web search and long-term memory tools
- Local inference using vLLM
- Support for multiple open-source LLMs
- GPT-4.1-based evaluation pipeline
- Detailed latency, token, and reasoning statistics
- Automatic experiment checkpointing and resume support
- Support for MATH-500 and GAIA benchmark datasets

---

## Requirements

- Python 3.10+
- CUDA-enabled GPU(s)
- vLLM
- Azure OpenAI or OpenAI-compatible judge endpoint
- HuggingFace Datasets

Install dependencies:

```bash
pip install -r requirements.txt
```

Install vLLM:

```bash
pip install vllm
```

---

## Clone Repository

```bash
git clone https://github.com/juealcs/Reasoning_Agentic_AI.git
cd Reasoning_Agentic_AI
```

---

# Start Tool Model

Run the tool model on **GPU 0**.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen35-4b \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

---

# Start Final Model

Run **one** final model server on **GPU 1**.

## DeepSeek-R1-Distill-Qwen-7B

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name deepseek-r1-7b \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --enforce-eager
```

## Llama-3.1-8B-Instruct

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name llama31-8b \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

---

# Run Benchmark

## MATH-500

```bash
python main.py \
  --tool_model_choice qwen35-4b \
  --model_choice llama31-8b \
  --tool_base_url http://localhost:8000/v1 \
  --final_base_url http://localhost:8001/v1 \
  --api_key EMPTY \
  --dataset_id HuggingFaceH4/MATH-500 \
  --split test \
  --max_samples 500 \
  --max_tokens 4096
```

## GAIA

```bash
python main.py \
  --tool_model_choice qwen35-4b \
  --model_choice llama31-8b \
  --tool_base_url http://localhost:8000/v1 \
  --final_base_url http://localhost:8001/v1 \
  --api_key EMPTY \
  --dataset_id gaia-benchmark/GAIA \
  --dataset_config 2023_all \
  --split validation \
  --prompt_field Question \
  --answer_field "Final answer" \
  --max_samples 165 \
  --max_tokens 4096
```

---

# Configure Azure OpenAI Judge

```bash
export AZURE_OPENAI_API_KEY="<YOUR_API_KEY>"
export AZURE_OPENAI_BASE_URL="<YOUR_AZURE_ENDPOINT>"
export AZURE_OPENAI_DEPLOYMENT="gpt-4.1"
```

---

# Evaluate Results

```bash
python judge.py \
  --input_file file_path \
  --judge_model gpt-4.1
```

---

# Output Structure

```text
agentic_ai_benchmark/
└── vanilla/
    └── tool_qwen35-4b/
        └── final_llama31-8b/
            └── HuggingFaceH4_MATH_500/
                ├── *_latest.json
                ├── *_summary.json
                ├── *_memory.sqlite
                ├── *_checkpoints.sqlite
                ├── judge_result_*.json
                └── judge_result_*_summary.json
```

For GAIA, the dataset folder will be similar to:

```text
agentic_ai_benchmark/
└── vanilla/
    └── tool_qwen35-4b/
        └── final_llama31-8b/
            └── gaia_benchmark_GAIA_2023_all/
                ├── *_latest.json
                ├── *_summary.json
                ├── *_memory.sqlite
                ├── *_checkpoints.sqlite
                ├── judge_result_*.json
                └── judge_result_*_summary.json
```

---

# Supported Models

| Model Choice | HuggingFace Model |
|--------------|-------------------|
| qwen35-4b | Qwen/Qwen3.5-4B |
| llama31-8b | meta-llama/Llama-3.1-8B-Instruct |
| deepseek-r1-7b | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |

---

# Supported Datasets

| Dataset | Dataset ID | Config | Split | Prompt Field | Answer Field |
|---------|------------|--------|-------|--------------|--------------|
| MATH-500 | HuggingFaceH4/MATH-500 | None | test | auto | auto |
| GAIA | gaia-benchmark/GAIA | 2023_all | validation | Question | Final answer |

---

# Recorded Metrics

The benchmark automatically records:

- Accuracy
- Input tokens
- Output tokens
- Total tokens
- Estimated reasoning tokens
- Token-limit failures
- Tool calls
- Tool execution time
- Final inference time
- End-to-end latency
- Memory operations
- Web search usage
- Over-reasoning
- Under-reasoning
- Complete execution trace

---

# Citation

If you use this repository in your research, please cite:

```bibtex
will be updated soon
```
