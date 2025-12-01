# python ./hal-detector/uncertainty_quantification/hallucination_detection_gpt4.py \
#  --hallucination_rate_output_file ./playground/vqav2/uncertainty_result/gpt4o-hallucination.jsonl \
#  --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-gpt4o.jsonl \
#  --token_entropy_file ./playground/vqav2/uncertainty_result/gpt4o-token_entropy.jsonl \
#  --output_file ./playground/vqav2/uncertainty_result/gpt4o-reweigh_token_entropy.jsonl \
#  --token_file ./playground/vqa_v2/answers/gpt4o_val2014_answers.jsonl \
#  --model_path ./weights/mPLUG-Owl3-7B-241101

python ./hal-detector/uncertainty_quantification/se_hallucination_detection_gpt4-wo.py \
 --hallucination_rate_output_file ./playground/Bingo_benchmark/uncertainty_result/ubder.jsonl \
 --importance_file ./playground/Bingo_benchmark/semantic_locate_result/semantic_locate_result-under.jsonl \
 --token_entropy_file ./playground/Bingo_benchmark/uncertainty_result/under-token_entropy.jsonl \
 --output_file ./playground/Bingo_benchmark/uncertainty_result/under-reweigh_token_entropy.jsonl \
 --token_file ./playground/Bingo_benchmark/answers/answer-gpt4-bingo.jsonl \
 --model_path gpt4


# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl \
#  --file_2 ./playground/vqav2/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --output_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination12.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/vqav2/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#   --hallucination_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination12.jsonl


# python ./hal-detector/uncertainty_quantification/hallucination_detection.py \
#  --hallucination_rate_output_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl \
#  --importance_file ./playground/vqav2/semantic_locate_result/semantic_locate_result-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --token_entropy_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-token_entropy.jsonl \
#  --output_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-reweigh_token_entropy1.jsonl \
#  --token_file ./playground/vqav2/answers/answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --model_path ./weights/mPLUG-Owl3-7B-241101

# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl \
#  --file_2 ./playground/vqav2/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --output_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination12.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/vqav2/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#   --hallucination_file ./playground/vqav2/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination12.jsonl
