# python ./hal-detector/uncertainty_quantification/se_hallucination_detection.py \
#  --hallucination_output_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-7b-hallucination2.jsonl \
#  --importance_file ./playground/Bingo_benchmark/semantic_locate_result/semantic_locate_result-llava-v1.5-7b-bingo.jsonl \
#  --token_entropy_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-7b-token_entropy.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-7b-reweigh_token_entropy2.jsonl \
#  --token_file ./playground/Bingo_benchmark/answers/answer-llava-v1.5-7b-bingo.jsonl \
#  --model_path ./weights/llava-v1.5-7b

# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-7b-hallucination2.jsonl \
#  --file_2 ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-llava-v1.5-7b-bingo.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-7b-hallucination22.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-llava-v1.5-7b-bingo.jsonl \
#   --hallucination_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-7b-hallucination22.jsonl

# python ./hal-detector/uncertainty_quantification/se_hallucination_detection.py \
#  --hallucination_output_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-13b-hallucination2.jsonl \
#  --importance_file ./playground/Bingo_benchmark/semantic_locate_result/semantic_locate_result-llava-v1.5-13b-bingo.jsonl \
#  --token_entropy_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-13b-token_entropy.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-13b-reweigh_token_entropy2.jsonl \
#  --token_file ./playground/Bingo_benchmark/answers/answer-llava-v1.5-13b-bingo.jsonl \
#  --model_path ./weights/llava-v1.5-13b

# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-13b-hallucination2.jsonl \
#  --file_2 ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-llava-v1.5-13b-bingo.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-13b-hallucination22.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-llava-v1.5-13b-bingo.jsonl \
#   --hallucination_file ./playground/Bingo_benchmark/uncertainty_result/llava-v1.5-13b-hallucination22.jsonl


# python ./hal-detector/uncertainty_quantification/hallucination_detection.py \
#  --hallucination_rate_output_file ./playground/Bingo_benchmark/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl \
#  --importance_file ./playground/Bingo_benchmark/semantic_locate_result/semantic_locate_result-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --token_entropy_file ./playground/Bingo_benchmark/uncertainty_result/mPLUG-Owl3-7B-241101-token_entropy.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/mPLUG-Owl3-7B-241101-reweigh_token_entropy1.jsonl \
#  --token_file ./playground/Bingo_benchmark/answers/answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --model_path ./weights/mPLUG-Owl3-7B-241101

# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/Bingo_benchmark/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl \
#  --file_2 ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination12.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-bingo.jsonl \
#   --hallucination_file ./playground/Bingo_benchmark/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination12.jsonl

# python ./hal-detector/uncertainty_quantification/se_hallucination_detection.py \
#  --hallucination_output_file ./playground/Bingo_benchmark/uncertainty_result/Llama-3.2-Vision-Instruct-hallucination2.jsonl \
#  --importance_file ./playground/Bingo_benchmark/semantic_locate_result/semantic_locate_result-Llama-3.2-11B-Vision-Instruct-bingo.jsonl \
#  --token_entropy_file ./playground/Bingo_benchmark/uncertainty_result/Llama-3.2-11B-Vision-Instruct-token_entropy.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/Llama-3.2-Vision-Instruct-reweigh_token_entropy2.jsonl \
#  --token_file ./playground/Bingo_benchmark/answers/answer-Llama-3.2-11B-Vision-Instruct-bingo.jsonl \
#  --model_path ./weights/Llama-3.2-11B-Vision-Instruct

# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/Bingo_benchmark/uncertainty_result/Llama-3.2-Vision-Instruct-hallucination2.jsonl \
#  --file_2 ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-Llama-3.2-11B-Vision-Instruct-bingo.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/Llama-3.2-Vision-Instruct-hallucination22.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-Llama-3.2-11B-Vision-Instruct-bingo.jsonl \
#   --hallucination_file ./playground/Bingo_benchmark/uncertainty_result/Llama-3.2-Vision-Instruct-hallucination22.jsonl


python ./hal-detector/uncertainty_quantification/se_hallucination_detection_gpt4.py \
 --hallucination_rate_output_file ./playground/Bingo_benchmark/uncertainty_result/gpt4-hallucination-under.jsonl \
 --importance_file ./playground/Bingo_benchmark/semantic_locate_result/semantic_locate_result-gpt4o.jsonl \
 --token_entropy_file ./playground/Bingo_benchmark/uncertainty_result/gpt4-token_entropy-under.jsonl \
 --output_file ./playground/Bingo_benchmark/uncertainty_result/gpt4-reweigh_token_entropy-under.jsonl \
 --token_file ./playground/Bingo_benchmark/answers/answer-gpt4-bingo.jsonl \
 --model_path gpt4

# python ./hal-detector/experiment/delet.py \
#  --file_1 ./playground/Bingo_benchmark/uncertainty_result/gpt4-hallucination2.jsonl \
#  --file_2 ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-gpt4-bingo.jsonl \
#  --output_file ./playground/Bingo_benchmark/uncertainty_result/gpt4-hallucination22.jsonl

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Bingo_benchmark/gpt4_ground_truth/ground_truth-answer-gpt4-bingo.jsonl \
#   --hallucination_file ./playground/Bingo_benchmark/uncertainty_result/gpt4-hallucination22.jsonl
