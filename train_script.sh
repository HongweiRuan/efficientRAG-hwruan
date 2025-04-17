# 2.1 Query Decompose
python src/data_synthesize/query_decompose.py \
    --dataset hotpotQA \
    --split train \
    --model qwen \
    --max_samples 10

# 2.2 Token Labeling
# 2.2.1 Token Labeling
python src/data_synthesize/token_labeling.py \
    --dataset hotpotQA \
    --split train \
    --model qwen \
    --max_samples 10

# 2.2.2 Token Extraction
python src/data_synthesize/token_extraction.py \
    --data_path data/synthesized_token_labeling/hotpotQA/train.jsonl \
    --save_path data/token_extracted/hotpotQA/train.jsonl \
    --verbose

# 2.3 Next Query Filtering
python src/data_synthesize/next_hop_query_construction.py \
    --dataset hotpotQA \
    --split train \
    --model qwen 

python src/data_synthesize/next_hop_query_filtering.py \
    --data_path data/synthesized_next_query/hotpotQA/train.jsonl \
    --save_path data/next_query_extracted/hotpotQA/train.jsonl \
    --verbose
    
# 2.4 Negative Sampling
python src/data_synthesize/negative_sampling.py \
    --dataset hotpotQA \
    --split train \
    --retriever contriever

# 2.4.2 Negative Sampling Labeled
python src/data_synthesize/negative_sampling_labeled.py \
    --dataset hotpotQA \
    --split train \
    --model llama

# 2.4.3 Negative Token Extraction
python src/data_synthesize/negative_token_extraction.py \
    --dataset hotpotQA \
    --split train \
    --verbose