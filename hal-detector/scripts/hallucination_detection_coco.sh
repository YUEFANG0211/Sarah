python ./hal-detector/uncertainty_quantification/hallucination_detection_gpt4.py \
 --hallucination_rate_output_file ./playground/Coco_caption/uncertainty_result/gpt4-hallucination1.jsonl \
 --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-gpt4-cococaption.jsonl \
 --token_entropy_file ./playground/Coco_caption/uncertainty_result/gpt4-token_entropy.jsonl \
 --output_file ./playground/Coco_caption/uncertainty_result/gpt4-reweigh_token_entropy.jsonl \
 --token_file ./playground/Coco_caption/answers/answer-gpt4-cococaption.jsonl \
 --model_path gpt4

python ./hal-detector/experiment/eval_positive.py \
  --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/ground_truth-answer-gpt4-cococaption.jsonl \
  --hallucination_file ./playground/Coco_caption/uncertainty_result/gpt4-hallucination1.jsonl




# python ./hal-detector/uncertainty_quantification/hallucination_detection.py \
#  --hallucination_rate_output_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-hallucination1.jsonl \
#  --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-llava-v1.5-7b-cococaption.jsonl \
#  --token_entropy_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-token_entropy.jsonl \
#  --output_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-reweigh_token_entropy.jsonl \
#  --token_file ./playground/Coco_caption/answers/answer-llava-v1.5-7b-cococaption.jsonl \
#  --model_path ./weights/llava-v1.5-7b

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/ground_truth-answer-llava-v1.5-7b-cococaption.jsonl \
#   --hallucination_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-hallucination1.jsonl


# python ./hal-detector/uncertainty_quantification/se_hallucination_detection.py \
#  --hallucination_output_file ./playground/Coco_caption/uncertainty_result/Llama-3.2-11B-Vision-Instruct-hallucination2.jsonl \
#  --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-Llama-3.2-11B-Vision-Instruct-cococaption.jsonl \
#  --token_entropy_file ./playground/Coco_caption/uncertainty_result/Llama-3.2-11B-Vision-Instruct-token_entropy.jsonl \
#  --output_file ./playground/Coco_caption/uncertainty_result/Llama-3.2-11B-Vision-Instruct-reweigh_token_entropy2.jsonl \
#  --token_file ./playground/Coco_caption/answers/answer-Llama-3.2-11B-Vision-Instruct-cococaption.jsonl \
#  --model_path ./weights/Llama-3.2-11B-Vision-Instruct

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/ground_truth-answer-Llama-3.2-11B-Vision-cococaption-1.jsonl \
#   --hallucination_file ./playground/Coco_caption/uncertainty_result/Llama-3.2-11B-Vision-Instruct-hallucination2.jsonl

# python ./hal-detector/uncertainty_quantification/hallucination_detection.py \
#  --hallucination_rate_output_file ./playground/Coco_caption/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl \
#  --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-mPLUG-Owl3-7B-241101-cococaption.jsonl \
#  --token_entropy_file ./playground/Coco_caption/uncertainty_result/mPLUG-Owl3-7B-241101-token_entropy.jsonl \
#  --output_file ./playground/Coco_caption/uncertainty_result/mPLUG-Owl3-7B-241101-reweigh_token_entropy1.jsonl \
#  --token_file ./playground/Coco_caption/answers/answer-mPLUG-Owl3-7B-241101-cococaption.jsonl \
#  --model_path ./weights/mPLUG-Owl3-7B-241101

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/ground_truth-answer-mPLUG-Owl3-7B-241101-cococaption.jsonl \
#   --hallucination_file ./playground/Coco_caption/uncertainty_result/mPLUG-Owl3-7B-241101-hallucination1.jsonl

# python ./hal-detector/uncertainty_quantification/hallucination_detection.py \
#  --hallucination_rate_output_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-hallucination1.jsonl \
#  --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-llava-v1.5-7b-cococaption.jsonl \
#  --token_entropy_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-token_entropy.jsonl \
#  --output_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-reweigh_token_entropy1.jsonl \
#  --token_file ./playground/Coco_caption/answers/answer-llava-v1.5-7b-cococaption.jsonl \
#  --model_path ./weights/llava-v1.5-7b


# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/ground_truth-answer-llava-v1.5-7b-cococaption.jsonl \
#   --hallucination_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-7b-hallucination1.jsonl

# python ./hal-detector/uncertainty_quantification/se_hallucination_detection.py \
#  --hallucination_output_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-13b-hallucination2.jsonl \
#  --importance_file ./playground/Coco_caption/semantic_locate_result/semantic_locate_result-llava-v1.5-13b-cococaption.jsonl \
#  --token_entropy_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-13b-token_entropy.jsonl \
#  --output_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-13b-reweigh_token_entropy2.jsonl \
#  --token_file ./playground/Coco_caption/answers/answer-llava-v1.5-13b-cococaption.jsonl \
#  --model_path ./weights/llava-v1.5-13b

# python ./hal-detector/experiment/eval_positive.py \
#   --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/ground_truth-answer-llava-v1.5-13b-cococaption.jsonl \
#   --hallucination_file ./playground/Coco_caption/uncertainty_result/llava-v1.5-13b-hallucination2.jsonl