import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)



def main(model_name, output_filename, lora_path=None, is_local_model=False, gpu_id=None):
    """
    Runs the MATH-500 test set evaluation.
    """
    # ------------------------------------------------------------------
    # 【关键修改】将 HuggingFace 相关库的导入移到这里
    # 确保在 if __name__ == "__main__" 中设置好代理/镜像后，才加载这些库
    # ------------------------------------------------------------------
    print("Importing Hugging Face libraries...")
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    # ------------------------------------------------------------------

    # 1. Load MATH-500 test set
    print("Loading dataset...")
    ds = load_dataset("ricdomolm/MATH-500", split="test")
    prompts = ds["problem"]
    gold_answers = ds["answer"]

    # 2. Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    prompt_chats = [
        [
            {"role": "user", "content": p + system_instruction}
        ]
        for p in prompts
    ]

    prompt_strs = [
        tokenizer.apply_chat_template(
            conversation=prompt_chat,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        for prompt_chat in prompt_chats
    ]

    # 3. Create Transformers Model
    if is_local_model:
        print(f"Loading local fine-tuned model from: {model_name}")
    else:
        print(f"Loading base model: {model_name}")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 根据gpu_id设置device_map
    if gpu_id is not None:
        print(f"Using specified GPU: {gpu_id}")
        device_map = {"": int(gpu_id)}
    else:
        # 使用auto模式，但会受到CUDA_VISIBLE_DEVICES的限制
        device_map = "auto"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map
    )

    if lora_path:
        print(f"LoRA adapter specified. Loading from local path: {lora_path}")
        if not os.path.isdir(lora_path):
            raise ValueError(f"LoRA path '{lora_path}' is not a valid directory.")

        try:
            # 使用load_adapter方法代替PeftModel.from_pretrained
            # 这是更现代的方法，适用于较新的PEFT版本
            model.load_adapter(lora_path)
            print("Successfully loaded LoRA adapter.")
        except (AttributeError, TypeError) as e:
            print(f"Failed to load adapter directly: {e}. Trying PeftModel approach...")
            # 回退到PeftModel方法
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print("Successfully merged LoRA adapter into the base model.")
    else:
        print("No LoRA adapter specified, running the base model.")

    model.eval()

    generation_kwargs = {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
        "do_sample": True,
    }

    # 4. Generate in batches
    batch_size = 4
    results = []

    total_batches = (len(prompt_strs) + batch_size - 1) // batch_size

    for i in range(0, len(prompt_strs), batch_size):
        batch = prompt_strs[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for idx, gen_text in enumerate(generated_texts):
            orig_idx = i + idx
            results.append({
                "id": orig_idx,
                "prompt": prompts[orig_idx],
                "answer": gen_text,
                "gold": gold_answers[orig_idx]
            })
        print(f"Processed batch {i // batch_size + 1}/{total_batches}")

    # 5. Save to JSONL
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved generations to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MATH-500 evaluation on a specific GPU.")
    parser.add_argument("--model", type=str, required=True,
                        help="The Hugging Face model path.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path.")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional: Path to LoRA adapter.")
    parser.add_argument("--is_local_model", action="store_true",
                        help="Set if model is a local path.")

    # 新增 GPU 选择参数
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="Specific GPU ID to run on (e.g., '0', '1', '7').")

    # 代理与镜像设置
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com", help="HF Mirror Endpoint.")
    parser.add_argument("--proxy", type=str, default=None, help="Network Proxy.")

    args = parser.parse_args()

    # 【重要】设置 GPU 环境变量
    # 必须在加载任何模型相关库之前设置，否则 device_map="auto" 会占用所有卡
    if args.gpu_id is not None:
        print(f"Running on GPU: {args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # 设置网络环境变量
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy

    main(args.model, args.output_file, args.lora_path, args.is_local_model, args.gpu_id)