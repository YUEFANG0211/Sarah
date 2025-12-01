import argparse
import json 
import os
import re
from llava.model.builder import load_pretrained_model
from scipy.special import softmax
from llava.mm_utils import  get_model_name_from_path
from transformers import AutoProcessor
from modelscope import AutoTokenizer
from nltk.corpus import wordnet as wn
import math
from typing import List, Dict
from nltk import pos_tag
import numpy as np
from nltk.tokenize import word_tokenize
import torch
import torch.nn.functional as F
from tqdm import tqdm

def calculate_entropy(probabilities: List[float]) -> float:
    """计算一个概率分布的熵"""
    entropy = 0
    for p in probabilities:
        if p > 0 and p < 1:  # 排除概率为零的情况
            entropy -= p * math.log(p, 2)
    return entropy

def get_data(file_path: str, args) -> Dict[str, List[Dict[int, List[float]]]]: 
    model_path = os.path.expanduser(args.model_path)
    if args.model_path == './weights/mPLUG-Owl3-7B-241101':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif args.model_path == './weights/llava-v1.5-7b' or args.model_path == './weights/llava-v1.5-13b':
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    elif args.model_path == './weights/Llama-3.2-11B-Vision-Instruct':
        tokenizer = AutoProcessor.from_pretrained(model_path)
    token_probabilities = {}
    # 打开并读取文件
    with open(file_path, 'r', encoding='utf-8') as infile:
        # 使用 tqdm 包装文件读取过程，显示进度条
        for line in tqdm(infile, desc="Reading file", unit="lines"):
            # 解析每一行的JSON数据
            data = json.loads(line.strip())
            input_id = data.get('id', '') 
            if input_id not in token_probabilities:
                token_probabilities[input_id] = []
            # 提取"probabilities"字段
            for tokendata in data.get("token_data", []):
                probabilities = tokendata["probabilities"]
                tokens = tokendata["tokens"]
                cluster_idx = [-1] * len(tokens)
                cur_cluster_idx = 0
                original = tokenizer.decode(tokens[0])
                for i in range(0, len(tokens)):
                    original_word=tokenizer.decode(tokens[i])
                    relationship = get_relationship(original ,original_word)
                    if cluster_idx[i] == -1:
                        cluster_idx[i] = cur_cluster_idx
                    
                    if relationship == "neutral":
                        #print("neutral")
                        probabilities[i] = 0
                        continue
                    
                    for j in range(i+1, len(tokens)):
                        if cluster_idx[j] == -1:
                            replace_word=tokenizer.decode(tokens[j])
                            relationship = get_relationship(original_word ,replace_word)
                            
                            if relationship == "entail":
                                probabilities[i] += probabilities[j]
                                probabilities[j] = 0  # 符合聚类条件标注为 0，与输出语义近似，可纳入当前输出的概率
                                cluster_idx[j] = cluster_idx[i]
                    cur_cluster_idx += 1

                token_probabilities[input_id].append({
                    'token_id': tokendata['token_id'],
                    'probabilities': probabilities
                })
    return token_probabilities

def get_data_for_gpt4(file_path: str, args) -> Dict[str, List[Dict[int, List[float]]]]: 
    token_probabilities = {}
    # 打开并读取文件
    with open(file_path, 'r', encoding='utf-8') as infile:
        # 使用 tqdm 包装文件读取过程，显示进度条
        for line in tqdm(infile, desc="Reading file", unit="lines"):
            # 解析每一行的JSON数据
            data = json.loads(line.strip())
            input_id = data.get('id', '') 
            if input_id not in token_probabilities:
                token_probabilities[input_id] = []
            prob = data.get("prob", {})
            # 提取"probabilities"字段
            for tokendata in prob.get("content", []):
                probabilities = [logprob_item["logprob"] for logprob_item in tokendata.get("top_logprobs", [])]
                probabilities =  torch.tensor(probabilities)
                # 对 probabilities 进行 softmax 处理
                probabilities = F.softmax(probabilities, dim=-1).tolist()
                tokens = [token_item["token"] for token_item in tokendata.get("top_logprobs", [])]
                # print(probabilities)
                # print(tokens)
                cluster_idx = [-1] * len(tokens)
                cur_cluster_idx = 0
                original = tokens[0]
                for i in range(0, len(tokens)):
                    original_word=tokens[i]
                    relationship = get_relationship(original ,original_word)
                    if cluster_idx[i] == -1:
                        cluster_idx[i] = cur_cluster_idx
                    
                    if relationship == "neutral":
                        # print("neutral")
                        probabilities[i] = 0
                        continue
                    
                    for j in range(i+1, len(tokens)):
                        if cluster_idx[j] == -1:
                            replace_word=tokens[j]
                            relationship = get_relationship(original_word ,replace_word)
                            
                            if relationship == "entail":
                                probabilities[i] += probabilities[j]
                                probabilities[j] = 0  # 符合聚类条件标注为 0，与输出语义近似，可纳入当前输出的概率
                                cluster_idx[j] = cluster_idx[i]
                    cur_cluster_idx += 1
                #print(probabilities)

                token_probabilities[input_id].append({
                    'token_id': tokens[0],
                    'probabilities': probabilities
                })
    return token_probabilities


def get_word_pos(word):
    if not word:  # 检查输入是否为空
        return None  # 返回 None 或其他默认值
    # 将单词放入一个列表中，以便使用pos_tag函数
    tokens = word_tokenize(word)
    # 获取词性标注
    pos_tags = pos_tag(tokens)
    if not pos_tags:  # 检查词性标注结果是否为空
        return None
    # 返回第一个词的词性
    return pos_tags[0][1]

def check_pos_same(word1, word2):
    pos1 = get_word_pos(word1)
    pos2 = get_word_pos(word2)
    if pos1 is None or pos2 is None:  # 如果任一词性为空，返回 False 或处理错误
        return False
    if pos1 == pos2:
        return True
    else:
        return False

def get_relationship(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if re.sub(r'[^\w\s]', '', word1).strip().lower() == re.sub(r'[^\w\s]', '', word2).strip().lower():    
        return "entail"
    if check_pos_same(word1, word2):
        # 检查是否有反义关系
        for syn1 in synsets1:
            for syn2 in synsets2:
                    # 检查蕴含关系（hypernyms 或 hyponyms）
                    if syn1 in syn2.closure(lambda s: s.hyponyms()):
                        # 检查相似度
                        similarity = syn1.wup_similarity(syn2)
                        if similarity and similarity > 0.7:  # 设置相似度阈值
                            return "entail"
                    elif syn1 == syn2:
                        return "entail"
                   # 检查 syn2 是否是 syn1 的上位词
                    elif syn2 in syn1.closure(lambda s: s.hypernyms()):
                        # 检查相似度
                        similarity = syn1.wup_similarity(syn2)
                        if similarity and similarity > 0.7:  # 设置相似度阈值
                            return "entail"                     
        return "contract"
    else:
        #print("neutral")
        return "neutral"

def save_token_probabilities(token_probabilities: Dict[str, List[Dict[int, List[float]]]], output_file: str) -> None:
    """将 token_probabilities 保存到 JSONL 文件中"""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for input_id, token_data in token_probabilities.items():
            # 将每个 input_id 对应的 token_data 写入文件
            json.dump({"input_id": input_id, "token_data": token_data}, outfile, ensure_ascii=False)
            outfile.write('\n')
    
def calculate_token_entropies(input_file: str, information_file: str, output_file: str, args) -> None:
    """计算 input_file.jsonl 中每个 aligned_token_ids 中 token 的熵，并输出到 output_file.jsonl"""
    # 加载 token_id 对应的概率分布
    if args.model_path == 'gpt4':
        #print('gpt4!')
        token_probabilities = get_data_for_gpt4(information_file, args)
        #print(token_probabilities)
    else:
        token_probabilities = get_data(information_file, args)

    # 读取 input_file 的所有行
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 处理每一行
    valid_lines = []  # 保存有效的行
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            try:
                data = json.loads(line.strip())
                original_text_id = data.get("original_text_id", "")
                aligned_token_ids = data.get("aligned_token_ids", [])
                
                # 如果 original_text_id 存在于 token_probabilities 中，继续处理
                if original_text_id in token_probabilities:
                    aligned_token_entropy = []
                    # 对每个 aligned_token_id 计算熵
                    for token_id in aligned_token_ids:
                        try:
                            # 获取对应的 token_id 索引
                            token_data = token_probabilities[original_text_id]
                            #print(token_data)
                            token_info = token_data[token_id - 1]  # -1 因为 token_id 从 1 开始
                            #print(token_info)
                            probabilities = token_info["probabilities"]
                        except KeyError:
                            probabilities = None  # 如果 original_text_id 或 token_id 不存在，返回 None
                        except IndexError:
                            probabilities = None  # 如果 token_id 超出范围，返回 None
                       
                        if probabilities:
                            entropy = calculate_entropy(probabilities)
                            aligned_token_entropy.append(entropy)
                        else:
                            aligned_token_entropy.append(None)  # 如果没有找到对应的概率分布
                    
                    # 将熵值添加到数据中
                    data["aligned_token_entropy"] = aligned_token_entropy
                    data["max_entropy"] = max(aligned_token_entropy) if aligned_token_entropy else None
                    
                    # 写入有效行到 output_file
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    valid_lines.append(line)  # 保存有效的行

                else:
                    print(f"Warning: original_text_id {original_text_id} not found in token probabilities.")
                    valid_lines.append(line)  # 保存有效的行

            except Exception as e:
                #print(f"Error processing line: {line}\nError: {e}")
                valid_lines.append(line)
                # 不将该行写入 output_file

    # 重新写回 valid_lines 到 input_file 中，删除无效的行
    with open(input_file, 'w', encoding='utf-8') as infile:
        infile.writelines(valid_lines)  # 将有效行重新写回到 input_file



# 主程序部分，处理命令行参数
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Calculate token entropies from token probabilities")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSONL file containing claim texts")
    parser.add_argument('--information_file', type=str, required=True, help="Information JSONL file with token probabilities")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSONL file to save results")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    
    args = parser.parse_args()

    # 执行计算熵的函数
    calculate_token_entropies(args.input_file, args.information_file, args.output_file, args)