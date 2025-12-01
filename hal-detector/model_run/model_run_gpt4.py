import openai
import argparse
import os
import json
from typing import Dict, List
import base64
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torch
import tiktoken




class HallucinationCheck:
    """
    Extracts claims from the text of the model generation.
    """

    def __init__(
        self,
        language: str = "en",
        progress_bar: bool = False,
    ):
        super().__init__()
        self.language = language
        self.progress_bar = progress_bar
        # self.experiment_prompts = experiment_prompts

    def __call__(
        self,
        id: str="",
        image_file: str ="",
        instruction: str ="",
    ) -> Dict[str, List]:
        args = parse_args()
        chat_ask = instruction
        # 图片的存储路径，需要补充具体图片名称
        image_file = image_file.lstrip('/')
        
        image_folder = args.image_folder
        
        image_path = os.path.join(image_folder, image_file)
      
        result, prob = self.process_text_with_gpt4o(prompt = chat_ask, image_path = image_path )
        
        with open(args.output_file, 'a', encoding='utf-8') as file:
            data = {
                "id": id,
                "instruction": instruction,
                "text": result,
                "prob": prob,
            }
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')
        return result, prob
    
    # 将图片文件转换为 Base64 编码
    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_image
    
    def process_text_with_gpt4o(self, prompt="", image_path = ""):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
            openai.api_key = api_key
            base64_image = self.image_to_base64(image_path)
            messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant with the ability to engage in conversations with the user based on the given image. Please provide your response in two simple sentences.Do not include any special characters except for standard punctuation like periods (.), commas (,), and question marks (?). Please answer the following question in one word."},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"}}, 
                        ], 
                    }
                    ]
            response = openai.ChatCompletion.create(
               model="gpt-4o",
               temperature = 0,
               messages = messages,
               max_tokens=64,
               logprobs=True,
               top_logprobs = 5
            )
            reply = response['choices'][0]['message']['content']
            logprobs_obj = response['choices'][0]['logprobs']

            # 将 logprobs 转为概率，并按每个 token 的 top_k 归一化
            def convert_logprobs_to_probs(lp_obj):
                if not lp_obj or 'content' not in lp_obj:
                    return lp_obj
                out = {'content': []}
                for token_info in lp_obj.get('content', []):
                    if not token_info:
                        out['content'].append(token_info)
                        continue
                    top_items = token_info.get('top_logprobs', []) or []
                    probs = []
                    for item in top_items:
                        lp = item.get('logprob')
                        try:
                            p = float(torch.exp(torch.tensor(lp)).item()) if lp is not None else 0.0
                        except Exception:
                            p = 0.0
                        probs.append(p)
                    s = sum(probs) if probs else 0.0
                    norm_probs = [(p / s) if s > 0 else 0.0 for p in probs]

                    # 写回为概率字段
                    new_top = []
                    for item, p in zip(top_items, norm_probs):
                        new_top.append({
                            'token': item.get('token'),
                            'prob': p
                        })

                    new_entry = dict(token_info)
                    if 'logprob' in new_entry:
                        try:
                            new_entry['prob'] = float(torch.exp(torch.tensor(new_entry['logprob'])).item())
                        except Exception:
                            new_entry['prob'] = None
                        new_entry.pop('logprob', None)
                    new_entry['top_probs'] = new_top
                    new_entry.pop('top_logprobs', None)
                    out['content'].append(new_entry)
                return out

            prob = convert_logprobs_to_probs(logprobs_obj)
            return reply.strip(), prob
        except openai.error.OpenAIError as e:
            print(f"Error in OpenAI API call: {e}")
            return ""
    
# Function to parse arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Extract claims from text.")
    parser.add_argument('--input-file', type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument('--image-folder', type=str, required=True, help="Path to image JSONL file.")
    parser.add_argument('--output-file', type=str, required=True, help="Path to output JSONL file.")

    return parser.parse_args()

def main():
    args = parse_args()
    # Initialize the HallucinationChecker
    hallucination_check = HallucinationCheck()
    # 打开输入文件并逐行处理
    with open(args.input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 先读取所有行，方便后续使用进度条
        # 使用 tqdm 包装文件行的迭代，显示进度条
        for line in tqdm(lines, desc="Running gpt-4o", unit="line"):
            data = json.loads(line.strip())
            instruction = data.get("instruction","") 
            id = data.get("id","")
            image_file = data.get("image","")
            result = hallucination_check(id, image_file, instruction)
           
if __name__ == "__main__":
    main()

