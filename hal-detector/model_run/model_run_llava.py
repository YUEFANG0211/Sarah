import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import numpy as np


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # 禁用 torch 初始化
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

   # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r", encoding="utf-8")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        image_file = line["image"]
        instruction = line["instruction"]  # Renamed text to instruction
        cur_prompt = instruction
        if model.config.mm_use_im_start_end:
            instruction = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + 'Answer in one word.'+instruction
        else:
            instruction = DEFAULT_IMAGE_TOKEN + '\n'+ 'Answer in one word.' + instruction

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_file = image_file.lstrip('/')
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True,
                return_dict_in_generate=True, 
                output_scores=True)

        # 检查 output_ids 是否包含非整数值
        try:
            outputs = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)[0].strip()
        except ValueError as e:
            print(f"Error decoding output_ids: {e}")
            print("Output IDs contents:", output_ids)
            continue  # Skip this iteration if decoding fails

        # 获取 logits（从生成的输出中提取 logits）
        logits = torch.stack(output_ids.scores, dim=1)  # logits的形状是 [batch_size, seq_len, vocab_size]

        token_data = []  # 用于存储每个 token 和对应的概率
        
        # 为每个 token 计算概率分布并排序
        for i in range(logits.shape[1]):  # 遍历每个 token
            token_logits = logits[0, i, :].cpu().numpy()  # 获取当前 token 的 logits
            token_probs = torch.softmax(torch.tensor(token_logits), dim=-1).numpy()  # 计算概率分布

            # 选择最大的 n 个 token 和它们的概率
            n_best_tokens = np.argsort(token_probs)[::-1][:args.top_k]  # 获取 top_k 的最可能的 token
            n_best_probs = token_probs[n_best_tokens]
            # n_best_tokens_str = tokenizer.convert_ids_to_tokens(n_best_tokens)
            # print(n_best_tokens_str)
            
            # 将 token_id 和对应的 probabilities 存入 token_data
            token_data.append({
                "token_id": int(output_ids.sequences[0, i]),  # 获取当前 token 的 id
                "probabilities": n_best_probs.tolist(),  # 获取当前 token 的概率分布
                "tokens": n_best_tokens.tolist() # 获取当前 token 的分布
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
