import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import numpy as np
import argparse
import torch
from modelscope import AutoConfig, AutoModel, AutoTokenizer
from decord import VideoReader, cpu 

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # 禁用 torch 初始化
    model_path = os.path.expanduser(args.model_path)
    model_name = 'mPLUG-Owl3-7B-241101'
    model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16, trust_remote_code=True)
    _ = model.eval().cuda()
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)
 
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r", encoding="utf-8-sig")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        image_file = line["image"]
        image_file = image_file.lstrip('/')
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        
        instruction = line["instruction"]  # Renamed text to instruction
        cur_prompt = instruction
        
       
        messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with the ability to interpret images. Please answer the user's questions based on the image. Your response must be in English. Answers in one word."
    },
    {
        "role": "user",
        "content": f"<|image|> {cur_prompt}"
    },
    {
        "role": "assistant",
        "content": ""
    }
]
        inputs = processor(messages, images=[image], videos=None)

        inputs.to('cuda')
        kwargs = ({
            'tokenizer': tokenizer,
            'max_new_tokens':100,
            'top_k': 10,
            'decode_text': True,
        })
 
        with torch.inference_mode():
            output, output_ids  = model.generate(**inputs, **kwargs)
        print("output_ids keys:", output_ids.keys() if hasattr(output_ids, 'keys') else "output_ids is not a dict")
        print("output_ids type:", type(output_ids))
        if hasattr(output_ids, 'keys'):
            for key in output_ids.keys():
                print(f"  {key}: {type(output_ids[key])}")
        else:
            print("output_ids attributes:", dir(output_ids))
        input_ids = inputs["input_ids"]
        print(f"input_ids shape: {input_ids.shape}")
        print(f"output_ids.scores type: {type(output_ids.scores)}")
        if hasattr(output_ids, 'scores') and output_ids.scores is not None:
            print(f"output_ids.scores length: {len(output_ids.scores)}")
            if len(output_ids.scores) > 0:
                print(f"output_ids.scores[0] shape: {output_ids.scores[0].shape}")
        else:
            print("output_ids.scores is None or doesn't exist")
        
        try:
            # 直接使用所有生成的 logits，不进行切片
            logits = torch.stack(output_ids.scores, dim=1)  # [batch_size, seq_len, vocab_size]
            print(f"Full logits shape: {logits.shape}")
            print(f"Input sequence length: {input_ids.shape[1]}")
            print(f"Generated sequence length: {logits.shape[1]}")
            
            # 只取生成的部分（去掉输入部分）
            if logits.shape[1] > input_ids.shape[1]:
                logits = logits[:, input_ids.shape[1]:, :]
                print(f"Trimmed logits shape: {logits.shape}")
            else:
                print("Warning: No new tokens generated, using all logits")
        except Exception as e:
            print(f"Error creating logits: {e}")
            logits = None
        token_data = []  # 用于存储每个 token 和对应的概率
        # 为每个 token 计算概率分布并排序
        
        if logits is not None:
            print('计算概率')
            for i in range(logits.shape[1]): 
                print('开始遍历') # 遍历每个 token
                token_logits = logits[0, i, :].cpu().numpy()  # 获取当前 token 的 logits
                print(token_logits)
                token_probs = torch.softmax(torch.tensor(token_logits), dim=-1).numpy()  # 计算概率分布
                print(token_probs)
                # 选择最大的 n 个 token 和它们的概率
                n_best_tokens = np.argsort(token_probs)[::-1][:args.top_k]  # 获取 top_k 的最可能的 token
                n_best_probs = token_probs[n_best_tokens]
                # n_best_tokens_str = tokenizer.convert_ids_to_tokens(n_best_tokens)
                print(n_best_probs)
                
                # 将 token_id 和对应的 probabilities 存入 token_data
                # 注意：i 是相对于生成序列的索引，需要加上输入序列长度来获取正确的 token_id
                token_idx = input_ids.shape[1] + i
                token_data.append({
                    "token_id": int(output_ids.sequences[0, token_idx]),  # 获取当前 token 的 id
                    "probabilities": n_best_probs.tolist(),  # 获取当前 token 的概率分布
                    "tokens": n_best_tokens.tolist() # 获取当前 token 的分布
                })
                print(token_data)
        else:
            print("logits is None, skipping token_data generation")

        # 保存答案到结果文件
        ans_id = shortuuid.uuid()  # Use a unique ID for the answer
        ans_file.write(json.dumps({
            "id": line.get("id", shortuuid.uuid()),  # Renamed question_id to id
            "instruction": cur_prompt,  # Renamed text to instruction
            "text": output[0],
            "answer_id": ans_id,
            "model_id": model_name,# 获取当前 token 的分布
            "token_data": token_data  # 保存 token_id 和 probabilities
        }) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)