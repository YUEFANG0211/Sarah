import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
import numpy as np
import base64
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
    processor = AutoProcessor.from_pretrained(model_path)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r", encoding="utf-8")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):  
        instruction = line["instruction"]  # 用户输入的指令
        cur_prompt = instruction

        image_file = line["image"]
        image_file = image_file.lstrip('/')
        image_folder = args.image_folder
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path) 
        
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant with the ability to engage in conversations with the user based on the given image in two simple setences. <|eot_id|><|start_header_id|>user<|end_header_id|><|image|>{cur_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are a helpful assistant with the ability to engage in conversations with the user based on the given image. 
        Please provide your response in one word. 
        Do not include any special characters except for standard punctuation like periods (.), commas (,), and question marks (?).
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        <|image|>{cur_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

        kwargs = ({
                    'max_new_tokens':1024,
                    'return_dict_in_generate': True, 
                    'output_scores': True,
                    'do_sample': False,
                    #'temperature': 1.0,
                    #'top_p': 0.95
                })

        # 生成输出
        with torch.no_grad():
            output_ids = model.generate(**inputs, **kwargs)
       
        generated_sequences = output_ids.sequences
        # 检查 output_ids 是否包含非整数值
        try:
            input_ids = inputs["input_ids"]
            # output = generated_sequences[0][:,input_ids.shape[1]:]
            output = generated_sequences[0][input_ids.shape[1]:]
            outputs = processor.decode(output)
            # print(outputs)
        except ValueError as e:
            print(f"Error decoding output_ids: {e}")
            print("Output IDs contents:", output_ids)
            continue  # Skip this iteration if decoding fails

        # 获取 logits（从生成的输出中提取 logits）
        logits = torch.stack(output_ids.scores, dim=1)  # logits的形状是 [batch_size, seq_len, vocab_size]

        token_data = []  # 用于存储每个 token 和对应的概率

        # 为每个 token 计算概率分布并排序
        for i in range(logits.shape[1]): 
            token_logits = logits[0, i, :].cpu().numpy()  # 获取当前 token 的 logits
            token_probs = torch.softmax(torch.tensor(token_logits), dim=-1).numpy()  # 计算概率分布

            # 选择最大的 n 个 token 和它们的概率
            n_best_tokens = np.argsort(token_probs)[::-1][:args.top_k]  # 获取 top_k 的最可能的 token
            n_best_probs = token_probs[n_best_tokens]

            # 将 token_id 和对应的 probabilities 存入 token_data
            token_data.append({
                "token_id": int(output_ids.sequences[0, i]),  # 获取当前 token 的 id
                "probabilities": n_best_probs.tolist(),  # 获取当前 token 的概率分布
                "tokens": n_best_tokens.tolist()  # 获取当前 token 的分布
            })

        # 保存答案到结果文件
        ans_id = shortuuid.uuid()  # Use a unique ID for the answer
        ans_file.write(json.dumps({
            "id": line.get("id", shortuuid.uuid()),  # Renamed question_id to id
            "instruction": cur_prompt,  # Renamed text to instruction
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "token_data": token_data  # 保存 token_id 和 probabilities
        }) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
