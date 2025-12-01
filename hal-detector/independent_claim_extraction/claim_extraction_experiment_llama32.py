import json
import json
import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import openai
from claim_level_prompt import CLAIM_EXTRACTION_PROMPTS, MATCHING_PROMPTS, CLAIM_EXTRACTION_PROMPTS_2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from transformers import MllamaForConditionalGeneration, AutoProcessor

@dataclass
class Claim:
    claim_text: str
    sentence: str
    aligned_token_ids: List[int]
    claim_id: int


class ClaimsExtractor:
    """
    Extracts claims from the text of the model generation.
    """

    def __init__(
        self,
        sent_separators: str = ".?!。？！\n",
        language: str = "en",
        progress_bar: bool = False,
        extraction_prompts: Dict[str, str] = CLAIM_EXTRACTION_PROMPTS,
        matching_prompts: Dict[str, str] = MATCHING_PROMPTS,
        n_threads: int = 1,
        # first_prompt = True

    ):
        super().__init__()
        self.language = language
        self.sent_separators = sent_separators
        self.progress_bar = progress_bar
        self.extraction_prompts = extraction_prompts
        self.matching_prompts = matching_prompts
        self.n_threads = n_threads
        self.claim_id_counter = 0  # Initialize claim_id_counter
        # self.first_prompt = first_prompt

    def __call__(
        self,
        dependencies: Dict[str, object],
        model_path: str,
    ) -> Dict[str, List]:
        """
        Extracts the claims out of each generation text from the input JSONL file.
        """
        
        # Reset claim_id_counter for the new image (i.e., new greedy_texts)
        self.claim_id_counter = 0

        texts = dependencies["greedy_texts"]
        greedy_texts = dependencies["greedy_texts"]
        greedy_texts = [greedy_texts]
        greedy_tokens = dependencies["greedy_tokens"]
        greedy_tokens = [greedy_tokens]
        claims: List[List[Claim]] = []
        claim_texts_concatenated: List[str] = []
        claim_input_texts_concatenated: List[str] = []
        processor = AutoProcessor.from_pretrained(model_path)
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            claims = list(
                tqdm(
                    executor.map(
                        self.claims_from_text,
                        greedy_texts,
                        greedy_tokens,
                        [processor] * len(greedy_texts),
                    ),
                    total=len(greedy_texts),
                    desc="Extracting claims",
                    disable=not self.progress_bar,
                )
            )

        for c in claims:
            for claim in c:
                claim_texts_concatenated.append(claim.claim_text)
                claim_input_texts_concatenated.append(texts[0])

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def claims_from_text(self, text: str, tokens: List[int], processor) -> List[Claim]:
        sentences = []
        for s in re.split(f"[{self.sent_separators}]", text):
            if len(s) > 0 :
                sentences.append(s)
        
        if len(text) > 0 and text[-1] not in self.sent_separators:
            sentences = sentences[:-1]

        sent_start_token_idx, sent_end_token_idx = 0, 0
        sent_start_idx, sent_end_idx = 0, 0
        claims = []
       
        for s in sentences:
            # print(s)
            while not text[sent_start_idx:].startswith(s):
                sent_start_idx += 1
            while not text[:sent_end_idx].endswith(s):
                sent_end_idx += 1
            
            while len(processor.decode(tokens[:sent_start_token_idx])) < sent_start_idx:
                sent_start_token_idx += 1
            while len(processor.decode(tokens[:sent_end_token_idx])) < sent_end_idx:
                sent_end_token_idx += 1
            #print(processor.decode(tokens[sent_start_token_idx:sent_end_token_idx]) )
           # print(text[sent_start_idx:sent_end_idx])
            for c in self._claims_from_sentence(
                s, tokens[sent_start_token_idx:sent_end_token_idx], processor
            ):
                for i in range(len(c.aligned_token_ids)):
                    c.aligned_token_ids[i] += sent_start_token_idx
             
                c.claim_id = self.claim_id_counter  # Assign a unique claim_id
                self.claim_id_counter += 1  # Increment the counter
                claims.append(c)
        return claims

    def _claims_from_sentence(
        self,
        sent: str,
        sent_tokens: List[int],
        processor,
    ) -> List[Claim]:
        # self.first_prompt = True
        # if  self.first_prompt:
        chat_ask = CLAIM_EXTRACTION_PROMPTS["en"].format(sent=sent)
           #  self.first_prompt = False  # 设置为False，表示已经使用过第一次提示词
        # else:
            # chat_ask = CLAIM_EXTRACTION_PROMPTS_2["en"].format(sent=sent)
        # chat_ask = self.extraction_prompts[self.language].format(sent=sent)
        extracted_claims = process_text_with_gpt4o(lang="en", prompt=chat_ask)

        claims = []
        for claim_text in extracted_claims.split("\n"):
            if not claim_text.startswith("- "):
                continue
            if "there aren't any claims" in claim_text.lower():
                continue
            claim_text = claim_text[2:].strip()
      
            chat_ask = self.matching_prompts[self.language].format(
                sent=sent,
                claim=claim_text,
            )
            match_words = process_text_with_gpt4o(lang="en", prompt=chat_ask)
            # print(match_words)
         
            if self.language == "zh":
                match_words = match_words.strip().split(" ")
            else:
                match_words = match_words.strip().split(",")
            match_words = list(map(lambda x: x.strip(), match_words))
            if self.language == "zh":
                match_string = self._match_string_zh(sent, match_words)
            else:
                match_string = self._match_string(sent, match_words)
                # print(match_string)
         
            if match_string is None:
                continue
            aligned_token_ids = self._align(sent, match_string, sent_tokens, processor)
      
            if len(aligned_token_ids) == 0:
                continue
            claims.append(
                Claim(
                    claim_text=claim_text,
                    sentence=sent,
                    aligned_token_ids=aligned_token_ids,
                    claim_id=self.claim_id_counter,  # Assign claim_id
                )
            )
        return claims

    def _match_string(self, sent: str, match_words: List[str]) -> Optional[str]:
        """
        Greedily matching words from `match_words` to `sent`.
        Parameters:
            sent (str): sentence string
            match_words (List[str]): list of words from sent, in the same order they appear in it.
        Returns:
            Optional[str]: string of length len(sent), for each symbol in sent, '^' if it contains in one
                of the match_words if aligned to sent, ' ' otherwise.
                Returns None if matching failed, e.g. due to words in match_words, which are not present
                in sent, or of the words are specified not in the same order they appear in the sentence.
        Example:
            sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
            match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
            return '^^^^^ ^^^^^^^^                      ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        '
        """

        sent_pos = 0  # pointer to the sentence
        match_words_pos = 0  # pointer to the match_words list
        # Iteratively construct match_str with highlighted symbols, start with empty string
        match_str = ""
        while sent_pos < len(sent):
            # Check if current word cur_word can be located in sent[sent_pos:sent_pos + len(cur_word)]:
            # 1. check if symbols around word position are not letters
            check_boundaries = False
            if sent_pos == 0 or not sent[sent_pos - 1].isalpha():
                check_boundaries = True
            if check_boundaries and match_words_pos < len(match_words):
                cur_match_word = match_words[match_words_pos]
                right_idx = sent_pos + len(cur_match_word)
                if right_idx < len(sent):
                    check_boundaries = not sent[right_idx].isalpha()
                # 2. check if symbols in word position are the same as cur_word
                if check_boundaries and sent[sent_pos:].startswith(cur_match_word):
                    # Found match at sent[sent_pos] with cur_word
                    len_w = len(cur_match_word)
                    sent_pos += len_w
                    # Highlight this position in match string
                    match_str += "^" * len_w
                    match_words_pos += 1
                    continue
            # No match at sent[sent_pos], continue with the next position
            sent_pos += 1
            match_str += " "

        if match_words_pos < len(match_words):
            # Didn't match all words to the sentence.
            # Possibly because the match words are in the wrong order or are not present in sentence.
            return None

        return match_str

    def _match_string_zh(self, sent: str, match_words: List[str]) -> Optional[str]:
        # Greedily matching characters from `match_words` to `sent` for Chinese.
        # Returns None if matching failed, e.g. due to characters in match_words, which are not present
        # in sent, or if the characters are not in the same order they appear in the sentence.
        #
        # Example:
        # sent = '爱因斯坦也是一位和平主义者。'
        # match_words = ['爱因斯坦', '是', '和平', '主义者']
        # return '^^^^ ^  ^^^^'

        last = 0  # pointer to the sentence
        last_match = 0  # pointer to the match_words list
        match_str = ""

        # Iterate through each character in the input Chinese text
        for char in sent:
            # Check if the current character matches the next character in match_words[last_match]
            if last_match < len(match_words) and char == match_words[last_match][last]:
                # Match found, update pointers and match_str
                match_str += "^"
                last += 1
                if last == len(match_words[last_match]):
                    last = 0
                    last_match += 1
            else:
                # No match, append a space to match_str
                match_str += " "

        # Check if all characters in match_words have been matched
        if last_match < len(match_words):
            return None  # Didn't match all characters to the sentence

        return match_str
    
    def _align(
        self,
        sent: str,
        match_str: str,
        sent_tokens: List[int],
        tokenizer,
    ) -> List[int]:
        """
        Identifies token indices in `sent_tokens` that align with matching characters (marked by '^')
        in `match_str`. All tokens, which textual representations intersect with any of matching
        characters, are included. Partial intersections should be uncommon in practice.

        Args:
            sent: the original sentence.
            match_str: a string of the same length as `sent` where '^' characters indicate matches.
            sent_tokens: a list of token ids representing the tokenized version of `sent`.
            tokenizer: the tokenizer used to decode tokens.

        Returns:
            A list of integers representing the indices of tokens in `sent_tokens` that align with
            matching characters in `match_str`.
        """
        sent_pos = 0
        cur_token_i = 0
        # Iteratively find position of each new token.
        aligned_token_ids = []
        while sent_pos < len(sent) and cur_token_i < len(sent_tokens):
            cur_token_text = tokenizer.decode(sent_tokens[cur_token_i])
            # Try to find the position of cur_token_text in sentence, possibly in sent[sent_pos]
            if len(cur_token_text) == 0:
                # Skip non-informative token
                cur_token_i += 1
                continue
            if sent[sent_pos:].startswith(cur_token_text):
                # If the match string corresponding to the token contains matches, add to answer
                if any(
                    t == "^"
                    for t in match_str[sent_pos : sent_pos + len(cur_token_text)]
                ):
                    aligned_token_ids.append(cur_token_i)
                cur_token_i += 1
                sent_pos += len(cur_token_text)
            else:
                # Continue with the same token and next position in the sentence.
                sent_pos += 1
        return aligned_token_ids

def process_text_with_gpt4o(lang="en", prompt=""):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=1024
        )
        reply = response['choices'][0]['message']['content']
        return reply.strip()
    except openai.error.OpenAIError as e:
        print(f"Error in OpenAI API call: {e}")
        return ""


def save_processed_claims(file_path, claims, id, greedy_texts, instruction):
    with open(file_path, 'a', encoding='utf-8') as file:
        for claim in claims:
            claim_data = {
                "claim_text": claim.claim_text,
                "aligned_token_ids": claim.aligned_token_ids,
                "original_text_id": id,
                "instruction": instruction,
                "original_text": greedy_texts,
                "claim_id": claim.claim_id
            }
            json.dump(claim_data, file, ensure_ascii=False)
            file.write('\n')


def load_dependencies_from_jsonl(line: str) -> Dict[str, object]:
    # 用于存储提取的 "greedy_texts" 和 "greedy_tokens"
    greedy_texts = ''
    greedy_tokens = []
    id = ''
    instruction = ''
    
    # 解析每一行 JSON 数据
    data = json.loads(line.strip())  # 去掉每行两端的空白字符并解析为字典
    
    # 获取 "text" 和 "id" 对应的值
    greedy_texts = data.get("text", "")  # "text" 对应的值是字符串
    id = data.get("id", "")
    instruction = data.get("instruction","")
    
    # 获取 "token_data" 列表并提取所有 "token_id" 字段
    token_data = data.get("token_data", [])  # "token_data" 是包含字典的列表
    # greedy_tokens = [item.get("token") for item in token_data if "token" in item]  # 提取每个字典中的 "token_id"
    greedy_tokens = [item["tokens"][0] for item in token_data if "tokens" in item]
    
    # 构造并返回 dependencies 字典
    dependencies = {
        "greedy_texts": greedy_texts,
        "greedy_tokens": greedy_tokens,
        "id": id,
        "instruction": instruction
    }
    
    return dependencies

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


# Function to parse arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Extract claims from text.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument('--n_threads', type=int, default=1, help="Number of threads for parallel processing.")
    parser.add_argument('--language', type=str, default="en", help="Language of the input texts.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()

    claims_extractor = ClaimsExtractor(
        language=args.language,
        n_threads=args.n_threads,
        
    )
    input_file = args.input_file
    output_file = args.output_file

    # 打开输入文件并逐行处理
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 先读取所有行，方便后续使用进度条
        # 使用 tqdm 包装文件行的迭代，显示进度条
        for i, line in enumerate(tqdm(lines, desc="Processing claims", unit="line"), start=0):
            # 读取当前行的数据
            dependencies = load_dependencies_from_jsonl(line.strip())  # 每次读取一行并处理
            # Ensure that model_path is properly passed as part of the arguments
     
            result = claims_extractor(dependencies, model_path=args.model_path)
            # Save the processed claims to the output file
            for claims_list in result['claims']:
                save_processed_claims(output_file, claims_list, dependencies["id"], dependencies["greedy_texts"], dependencies["instruction"],)

if __name__ == "__main__":
    main()

