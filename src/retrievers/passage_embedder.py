import argparse
import os
import pickle
import sys
import random

from embeddings import Embedder, ModelTypes

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from retrievers.utils.utils import load_passages


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--passages",
        type=str,
        required=True,
        help="Path to passages (tsv or jsonl file)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output file")
    parser.add_argument("--model_type", type=str, default="e5-base-v2", choices=list(ModelTypes.keys()))
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=int(2e6), help="passages per chunk")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of data to sample (0.0-1.0)")
    parser.add_argument("--sample_size", type=int, default=None, help="Fixed number of samples to use (overrides sample_ratio)")
    args = parser.parse_args()
    return args


def main(opts):
    embedder = Embedder(
        opts.model_type,
        opts.model_name_or_path,
        batch_size=opts.batch_size,
        chunk_size=opts.chunk_size,
        text_normalize=True,
    )
    print(f"Loading passages from {opts.passages}")
    data = load_passages(opts.passages)
    
    # Sample the data if needed
    if opts.sample_size is not None and opts.sample_size < len(data):
        print(f"Sampling {opts.sample_size} passages (out of {len(data)})")
        data = random.sample(data, opts.sample_size)
    elif opts.sample_ratio < 1.0:
        sample_size = int(len(data) * opts.sample_ratio)
        print(f"Sampling {sample_size} passages ({opts.sample_ratio:.2%} of {len(data)})")
        data = random.sample(data, sample_size)
    
    output_dir = opts.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for idx, (ids, embeddings) in embedder.embed_passages(data):
        output_file = os.path.join(output_dir, f"passages_{idx:02d}")
        with open(output_file, "wb") as f:
            pickle.dump((ids, embeddings), f)
        print(f"Save {len(ids)} embeddings to {output_file}")


if __name__ == "__main__":
    options = parse_args()
    main(options)
