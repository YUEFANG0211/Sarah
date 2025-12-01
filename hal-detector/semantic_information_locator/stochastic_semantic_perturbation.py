import spacy
import random
import json
import os  # 导入os模块来处理文件路径
import argparse
import math
from bert_score import score  # 导入BertScore计算

# 加载spaCy英文模型
nlp = spacy.load("en_core_web_sm")

# 读取并解析UD English语料库（.conllu格式）
def load_ud_data(file_path):
    word_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                word = parts[1]
                pos = parts[3]
                if pos not in word_dict:
                    word_dict[pos] = []
                word_dict[pos].append(word)
    return word_dict

# 计算softmax
def softmax(x):
    """计算softmax"""
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x]

def generate_single_word_variations(doc, word_dict, id, claims_id):
    sentence_variations = []
    similarity_results = []  # 用于保存结果的列表
    word_importances = []  # 用于收集所有单词的重要性值
    
    for i, token in enumerate(doc):
        pos = token.pos_
        if pos in word_dict and len(word_dict[pos]) >= 10:
            # 随机选择5个替换单词
            replacement_words = random.sample(word_dict[pos], 10)
            scores_list = []
            modified_sentences = []  # 用于保存该位置替换后的句子
            for replacement in replacement_words:
                new_sentence = [t.text for t in doc]
                new_sentence[i] = replacement
                modified_sentence = ' '.join(new_sentence)
                modified_sentences.append(modified_sentence)
                
                # 计算BertScore
                original_sentence = doc.text
                _, _, scores = score([original_sentence], [modified_sentence], model_type='roberta-large', lang='en')
                avg_score = scores.mean().item()  # 获取平均相似度
                scores_list.append(avg_score)

            # 计算10个替换词BertScore的平均值
            total_avg_score = sum(scores_list) / len(scores_list)

            # 计算word importance
            word_importance = 1 - total_avg_score
            word_importances.append(word_importance)

            # 保存每个结果到列表
            similarity_results.append({
                "original_text_id": id,
                "claims_id": claims_id,
                "pos": i,
                "original sentence": doc.text,
                "word importance": word_importance,
               # "hallucination_rate": 0,
               # "hallucinated": False  
            })
            
        else:
            # 如果没有可替换的词，保留原句
            new_sentence = [t.text for t in doc]
            sentence_variations.append(' '.join(new_sentence))

    if word_importances:
        min_importance = min(word_importances)
        max_importance = max(word_importances)
        print(min_importance,  max_importance)
    
    threshold = 0.4

    for i, result in enumerate(similarity_results):
       # print(i)
        normalized_word_importance = (word_importances[i] - min_importance) / (max_importance - min_importance)
        result["word importance"] = normalized_word_importance
        #result["hallucination_rate"] = normalized_word_importance * aligned_token_entropy[i]
       # if result["hallucination_rate"] > threshold:
           #  result["hallucinated"] = True
       # else:
         #   result["hallucination_rate"] = 0  # 如果没有对应的entropy值，默认为0

    return similarity_results
# 主程序
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Generate sentence variations and calculate BertScore")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSONL file with 'claim_text'")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSONL file to save results")
    parser.add_argument('--ud_file', type=str, required=True, help="UD English file path for word data")
    #parser.add_argument('--entropy_file', type=str, required=True, help="Token entropy JSONL file path")
    args = parser.parse_args()

    # 加载UD数据
    ud_data = load_ud_data(args.ud_file)

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    from tqdm import tqdm
    import json

    # 假设你的 nlp 和 generate_single_word_variations 函数已经定义

    with open(args.input_file, 'r', encoding='utf-8') as infile, \
        open(args.output_file, 'w', encoding='utf-8') as outfile:

        # 获取文件总行数，以便进度条能够计算进度
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # 重置文件指针
        # entropyfile.seek(0)  # 重置文件指针

        # 使用 tqdm 包装 zip 迭代器，设置描述信息并定义总行数以支持进度条
        # for line_infile in tqdm(zip(infile, entropyfile), total=total_lines, desc="Processing"):
        for line_infile in tqdm(infile, total=total_lines, desc="Processing"):
            claim_data = json.loads(line_infile.strip())  # 解析每一行来自 infile 的 JSON
            #entropy_data = json.loads(line_entropyfile.strip())  # 解析每一行来自 entropyfile 的 JSON

            input_sentence = claim_data.get("claim_text", "")
            id = claim_data.get("original_text_id", "")
            claims_id = claim_data.get("claim_id", "")
            #aligned_token_entropy = entropy_data.get("aligned_token_entropy", [])

            if input_sentence:  # 如果有claim_text字段
                doc = nlp(input_sentence)

                # 生成单词变体的句子
                similarity_results = generate_single_word_variations(doc, ud_data, id, claims_id)

                # 将每个处理的结果写入到输出文件
                for result in similarity_results:
                    # 将结果转换为JSON格式并写入文件
                    json.dump(result, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    outfile.flush()

        print(f"Processed results have been saved to {args.output_file}")
