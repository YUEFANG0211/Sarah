import json
from collections import defaultdict
import re
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from transformers import AutoTokenizer
from modelscope import AutoTokenizer
from transformers import AutoProcessor
import os

# 读取 semantic_locate_result-mPLUG-Owl3-7B-241101-bingo.jsonl 文件
def load_bingo_data(bingo_file):
    bingo_data = defaultdict(list)
    with open(bingo_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            key = (data["original_text_id"], data["claims_id"])
            bingo_data[key].append(data)
    return bingo_data

# 读取 mPLUG-Owl3-7B-241101-token_entropy.jsonl 文件
def load_token_entropy_data(token_entropy_file):
    token_entropy_data = []
    with open(token_entropy_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            token_entropy_data.append(data)
    return token_entropy_data

# 读取 token_file 文件并提取 token_id 
def load_token_data(token_file):
    token_data = {}
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            message = data.get("prob", {}).get("content",[])
            #print(token_data)
            #用于bingo
            #token_data[data["id"]] = [token["token"] for token in data["token_data"]]
            # 用于coco
            token_data[data["original_text_id"]] = [token["token"] for token in message]

    return token_data

# 根据 aligned_token_ids 找到对应的 token_id
def get_token_ids(token_data, original_text_id, aligned_token_ids):
    if original_text_id in token_data:
        tokens = token_data[original_text_id]
        return [tokens[token_idx-1] for token_idx in aligned_token_ids]
    return []

def decode_tokens(tokenizer, token_ids):
    # 使用 tokenizer 的 decode 方法将 token_id 解码为词汇
    tokens = []
    for token_id in token_ids:
        decoded_token = tokenizer.decode([token_id], skip_special_tokens=True)
        if decoded_token.strip() == '':  # 如果解码后的词汇为空，则跳过
            tokens.append('<UNK>')  # 使用一个默认值，如 '<UNK>' 表示未知词
        else:
            tokens.append(decoded_token)
    return tokens


def process_sentence(sentence):
    """
    处理包含LaTeX标记的句子，移除LaTeX公式并解析普通文本。
    """
    # 去除 LaTeX 格式的公式，如 \( 和 \)
    sentence = re.sub(r'\\\(.*?\\\)', '', sentence)  # 去除 \(...\) 形式的公式
    sentence = re.sub(r'\\text\{.*?\}', '', sentence)  # 去除 \text{...} 形式的文本
    
    # 处理后，拆分单词和标点符号
    return re.findall(r'\w+|[^\w\s]', sentence)

def filter_unwanted_characters(tokens):
    """
    过滤掉不需要的字符，如 '\\', '(', ')', '.', '_'
    """
    unwanted_characters = ['\\', '(', ')', '.', '_']
    return [token for token in tokens if token not in unwanted_characters]

def merge_time_tokens(tokens):
    """
    合并数字和标点符号，如12:00为一个单独的token
    """
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i].isdigit() and tokens[i + 1] == ':':
            merged_tokens.append(tokens[i] + tokens[i + 1] + tokens[i + 2])  # 合并如 '12:00'
            i += 3
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens


def calculate_reweigh_token_entropy(importance_data, token_entropy_data, token_data, tokenizer):
    results = []
    hallucination_data = defaultdict(list)  # 用于存储每个 original_text_id 的幻觉判断结果
    max_reweigh_entropy_data = defaultdict(float)  # 用于存储每个 original_text_id 的最大 reweigh_token_entropy

    for entry in token_entropy_data:
        # 定位到某一输出中的一个 claim
        key = (entry["original_text_id"], entry["claim_id"])
        if key not in importance_data:
            continue  

        # 获取对应的 token_id
        token_ids = get_token_ids(token_data, entry["original_text_id"], entry["aligned_token_ids"])

        # 解码 token_id 对应的词汇
        if tokenizer is None:
            aligned_tokens = token_ids
        else:
            aligned_tokens = decode_tokens(tokenizer, token_ids)

        # 获取 bingo 数据中的 original sentence 和 word importance
        importance_entries = importance_data[key]
        original_sentence = importance_entries[0]["original sentence"]
        original_sentence_list = process_sentence(original_sentence)
        
        # 过滤不需要的字符
        original_sentence_list = filter_unwanted_characters(original_sentence_list)
        aligned_tokens = filter_unwanted_characters(aligned_tokens)

        # 创建 word_importance_dict，按位置将 word importance 对应到具体的单词
        word_importance = []
        for importance_entry in importance_entries:
            word_importance.append(importance_entry["word importance"])

        # 比对词汇并计算 reweigh_token_entropy
        reweigh_token_entropy = []
        for token, token_entropy in zip(aligned_tokens, entry["aligned_token_entropy"]):
            try:
                word_pos = original_sentence_list.index(token.strip())
                importance = word_importance[word_pos]
                reweigh_token_entropy.append((token_entropy) * importance)
            except ValueError:
                reweigh_token_entropy.append(0)

        # 将结果保存到新的字典中
        result = entry.copy()
        result["reweigh_token_entropy"] = reweigh_token_entropy
        results.append(result)

        # 判断是否出现幻觉
        sum_entropy = sum(value for value in reweigh_token_entropy if value > 1)

        # 根据总和判断是否出现幻觉
        hallucinate = 1 if sum_entropy >= 1 else 0

        # 将该条数据添加到 hallucination_data 中
        hallucination_data[entry["original_text_id"]].append({
            "claim_id": entry["claim_id"],
            "hallucinate": hallucinate
        })

        # 更新最大 reweigh_token_entropy
        max_reweigh_entropy = max(reweigh_token_entropy, default=0)
        if max_reweigh_entropy > max_reweigh_entropy_data[entry["original_text_id"]]:
            max_reweigh_entropy_data[entry["original_text_id"]] = max_reweigh_entropy

    # 计算每个 original_text_id 的 hallucination_rate
    hallucination_rate_results = []
    for original_text_id, claims in hallucination_data.items():
        total_claims = len(claims)  # claim_id 的总数
        total_hallucinate = sum(claim["hallucinate"] for claim in claims)  # 出现幻觉的总次数
        hallucination_rate = total_hallucinate / total_claims if total_claims > 0 else 0  # 计算幻觉率

        # 获取最大 reweigh_token_entropy
        max_reweigh_entropy = max_reweigh_entropy_data[original_text_id]

        # 将结果添加到 hallucination_rate_results 中
        hallucination_rate_results.append({
            "original_text_id": original_text_id,
            "hallucination_rate": hallucination_rate,
            "total_hallucinate": total_hallucinate,
            "max_reweigh_token_entropy": max_reweigh_entropy  # 添加最大 reweigh_token_entropy
        })

    return results, hallucination_rate_results

# 保存结果到新的 JSONL 文件
def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

# def save_hallucination_rate_results(hallucination_rate_results, hallucination_rate_output_file):
#     with open(hallucination_rate_output_file, "w", encoding="utf-8") as f:
#         for result in hallucination_rate_results:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")

def save_hallucination_rate_results(hallucination_rate_results, hallucination_rate_output_file):
    with open(hallucination_rate_output_file, "w", encoding="utf-8") as f:
        for result in hallucination_rate_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

# 主函数
def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Process semantic and hallucination data.")
    parser.add_argument("--hallucination_rate_output_file", type=str, required=True, help="Path to output hallucination rate results")
    parser.add_argument("--importance_file", type=str, required=True, help="Path to the importance file")
    parser.add_argument("--token_entropy_file", type=str, required=True, help="Path to the token entropy file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output reweighed token entropy results")
    parser.add_argument("--token_file", type=str, required=True, help="Path to the token file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")

    args = parser.parse_args()

    # 加载数据
    importance_data = load_bingo_data(args.importance_file)
    token_entropy_data = load_token_entropy_data(args.token_entropy_file)
    token_data = load_token_data(args.token_file)
    
    model_path = os.path.expanduser(args.model_path)
    if args.model_path == './weights/mPLUG-Owl3-7B-241101':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif args.model_path == './weights/llava-v1.5-7b' or args.model_path == './weights/llava-v1.5-13b':
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    elif args.model_path == './weights/Llama-3.2-11B-Vision-Instruct':
        tokenizer = AutoProcessor.from_pretrained(model_path)
    elif args.model_path == 'gpt4':
        tokenizer = None

    # 计算 reweigh_token_entropy
    results, hallucination_rate_results = calculate_reweigh_token_entropy(importance_data, token_entropy_data, token_data, tokenizer)

    # 保存结果
    save_results(results, args.output_file)
    save_hallucination_rate_results(hallucination_rate_results, args.hallucination_rate_output_file)

if __name__ == "__main__":
    main()

