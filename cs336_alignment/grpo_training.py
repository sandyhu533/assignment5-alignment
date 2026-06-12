import argparse
import torch
import json
from cs336_alignment import checkpoint
from cs336_alignment import vllm_utils

def main(args):
    
    train = json.loads("data/gsm8k/train.jsonl")
    test = json.loads("data/gsm8k/test.jsonl")
    
    model, tokenizer = checkpoint.get_model_and_tokenizer(args.model, "cuda:0")
    server = vllm_utils.start_server(
        model_id=args.model,
        host="localhost",
        port=8000,
        gpu=1,
        seed=0,
        load_format="auto",
        logging_level="INFO",
        gpu_memory_utilization=0.85
    )
    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OLMo-2-0425-1B")
    args = parser.parse_args()
    main(args)