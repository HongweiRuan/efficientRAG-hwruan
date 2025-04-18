# EfficientRAG-official

<div align=center>
    <img src="static/bert_labeler.png" width=300px>
</div>

Code repo for EMNLP 2024 paper - **EfficientRAG: Efficient Retriever for Multi-Hop Question Answering**

Efficient RAG is a new framework to train Labeler and Filter to learn to conduct multi-hop RAG without multiple LLM calls.

## Updates

- 2024-09-12 open source the code
- 2025-03-04 release our data

## Setup

You can now download our synthesized data from this [link](https://box.nju.edu.cn/f/a86b512077c7489b8da3/).

You should unzip the `EfficientRAG.zip` file and place all the data under the `data` directory.
Within this directory, the `negative_sampling_extracted` folder contains our final synthesized data, which is referenced in [2.4 Negative Sampling](https://github.com/NIL-zhuang/EfficientRAG-official?tab=readme-ov-file#24-negative-sampling).
Additionally, the `efficient_rag` directory includes two folders: `labeler` and `filter`, which store the training data constructed for the model, as referenced in [2.5 Training Data](https://github.com/NIL-zhuang/EfficientRAG-official?tab=readme-ov-file#25-training-data).

### 1. Installation

You need to install PyTorch >= 2.1.0 first, and then install dependent Python libraries by running the command

```bash
pip install -r requirements.txt
```

You can also create a conda environment with python>=3.9

```bash
conda create -n <ENV_NAME> python=3.9 pip
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Preparation

1. Download the dataset from [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), [2WikiMQA](https://github.com/Alab-NII/2wikimultihop) and [MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue). Separate them as train, dev and test set, and then put them under `data/dataset`.

2. Download the retriever model [Contriever](https://huggingface.co/facebook/contriever-msmarco) and base model [DeBERTa](https://huggingface.co/microsoft/deberta-v3-large), put them under `model_cache`

```bash
 python download_models.py
```

1. Prepare the corpus by extract documents and construct embedding.

```bash
python src/retrievers/multihop_data_extractor.py --dataset hotpotQA
```

```bash
python src/retrievers/passage_embedder.py \
  --passages data/corpus/hotpotQA/corpus.jsonl \
  --output_dir data/corpus/hotpotQA/contriever \
  --model_type contriever \
  --model_name_or_path model_cache/facebook/contriever-msmarco/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa \
  --batch_size 32 \
  --sample_ratio 0.1
```

4. Deploy [LLaMA-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) with [vLLM](https://github.com/vllm-project/vllm) framework, and configure it in `src/language_models/llama.py`

OpenAI-compatible API Server

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --served-model-name llama3 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

### 2. Training Data Construction

We will use hotpotQA training set as an example. You could construct 2WikiMQA and MuSiQue in the same way.

#### 2.1 Query Decompose

```bash
python src/data_synthesize/query_decompose.py \
    --dataset hotpotQA \
    --split train \
    --model llama3
```

#### 2.2 Token Labeling

```bash
python src/data_synthesize/token_labeling.py \
    --dataset hotpotQA \
    --split train \
    --model llama3
```

use this command to install en_core_web_sm

```
python -m spacy download en_core_web_sm
```

```bash
python src/data_synthesize/token_extraction.py \
    --data_path data/synthesized_token_labeling/hotpotQA/train.jsonl \
    --save_path data/token_extracted/hotpotQA/train.jsonl \
    --verbose
```

#### 2.3 Next Query Filtering

```bash
python src/data_synthesize/next_hop_query_construction.py \
    --dataset hotpotQA \
    --split train \
    --model llama
```

```bash
python src/data_synthesize/next_hop_query_filtering.py \
    --data_path data/synthesized_next_query/hotpotQA/train.jsonl \
    --save_path data/next_query_extracted/hotpotQA/train.jsonl \
    --verbose
```

#### 2.4 Negative Sampling

```bash
python src/data_synthesize/negative_sampling.py \
    --dataset hotpotQA \
    --split train \
    --retriever contriever
```

```bash
python src/data_synthesize/negative_sampling_labeled.py \
    --dataset hotpotQA \
    --split train \
    --model llama
```

```bash
python src/data_synthesize/negative_token_extraction.py \
    --dataset hotpotQA \
    --split train \
    --verbose
```

#### 2.5 Training Data

```bash
python src/data_synthesize/training_data_synthesize.py \
    --dataset hotpotQA \
    --split train
```

## Training

Training Filter model

```bash
python src/efficient_rag/filter_training.py \
    --dataset hotpotQA \
    --save_path saved_models/filter
```

Training Labeler model

```bash
python src/efficient_rag/labeler_training.py \
    --dataset hotpotQA \
    --tags 2
```

## Inference

EfficientRAG retrieve procedure

```bash
python src/efficientrag_retrieve.py \
    --dataset hotpotQA \
    --retriever contriever \
    --labels 2 \
    --labeler_ckpt <<PATH_TO_LABELER_CKPT>> \
    --filter_ckpt <<PATH_TO_FILTER_CKPT>> \
    --topk 10 \
```

Use LLaMA-3-8B-Instruct as generator

```bash
python src/efficientrag_qa.py \
    --fpath <<MODEL_INFERENCE_RESULT>> \
    --model llama-8B \
    --dataset hotpotQA
```

## Citation

If you find this paper or code useful, please cite by:

```txt
@inproceedings{zhuang2024efficientrag,
  title={EfficientRAG: Efficient Retriever for Multi-Hop Question Answering},
  author={Zhuang, Ziyuan and Zhang, Zhiyang and Cheng, Sitao and Yang, Fangkai and Liu, Jia and Huang, Shujian and Lin, Qingwei and Rajmohan, Saravan and Zhang, Dongmei and Zhang, Qi},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={3392--3411},
  year={2024}
}
```
