import os
import openai
from experiment_prompt import SENTENCE_LEVEL_GPT_CHECK_PROMPT
import argparse
import json
from typing import Dict
from tqdm import tqdm
from pycocotools.coco import COCO

class HallucinationCheck:
    """
    Extracts claims from the text of the model generation.
    """

    def __init__(
        self,
        language: str = "en",
        progress_bar: bool = False,
        gpt_check_prompts: Dict[str, str] = SENTENCE_LEVEL_GPT_CHECK_PROMPT,
    ):
        super().__init__()
        self.language = language
        self.progress_bar = progress_bar
        self.gpt_check_prompts = gpt_check_prompts

    def __call__(
        self,
        id: str="",
        instruction: str ="",
        text: str ="",
        ground_truth: str ="",
    ) -> Dict[str, str]:
        chat_ask = self.gpt_check_prompts[self.language].format(
                instruction = instruction,
                ground_truth = ground_truth,
                text = text,
            )
        result = self.process_text_with_gpt4o(prompt=chat_ask)
        # 直接返回一个字典，而不是列表
        return {
                "id": id,
                "hallucination": result  
            }
    
    def process_text_with_gpt4o(self, prompt=""):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
            openai.api_key = api_key
            messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant capable of evaluating the factual correctness of claims based on both visual and textual data. You are expected to analyze the relationship between an image and a textual claim to identify if the claim is accurate."
                    },
                    {
                        "role": "user", 
                        "content": prompt,
                    }
                    ]
            response = openai.ChatCompletion.create(
               model="gpt-4o",
               messages = messages,
               max_tokens=10
            )
            reply = response['choices'][0]['message']['content']
            return reply.strip()
        except openai.error.OpenAIError as e:
            print(f"Error in OpenAI API call: {e}")
            return ""

# Function to parse arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Extract claims from text.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument('--groundtruth_file', type=str, required=True, help="Path to ground truth JSONL file.")
    return parser.parse_args()



def main():
    args = parse_args()
    # Initialize the HallucinationChecker
    hallucination_check = HallucinationCheck()
    coco = COCO(args.groundtruth_file)
    
    with open(args.input_file, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        
        for line_infile in tqdm(infile, desc="Processing", unit="lines"):
            lvlm_output = json.loads(line_infile.strip())  # 解析每一行来自 infile 的 JSON
            
            id = lvlm_output.get("id", "")
            instruction = lvlm_output.get("instruction", "")
            text = lvlm_output.get("text", "")
            img_id = int(id) if id and int(id) != 0 else None

            if img_id is not None:
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)
                gt = " ".join([ann["caption"] for ann in annotations[:5]])
                # gt = [ann["caption"] for ann in annotations]
                #print(gt)
            else:
                print("No valid image ID found.")
            
            # Ensure that model_path is properly passed as part of the arguments
            result = hallucination_check(id=id, instruction=instruction, text=text, ground_truth=gt)
            
            # 将每个处理的结果写入到输出文件
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write("\n")
            outfile.flush()  # 确保内容写入文件

if __name__ == "__main__":
    main()
