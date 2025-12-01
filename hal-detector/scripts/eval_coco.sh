
python ./hal-detector/experiment/eval.py \
  --ground_truth_file ./playground/Coco_caption/gpt4_ground_truth/vl-ground_truth-answer-llava7b-cococaption.jsonl \
  --hallucination_file ./playground/Coco_caption/uncertainty_result/vl-llava7b-coco.jsonl \
  --text_file  ./playground/Coco_caption/answers/answer-llava-v1.5-7b-cococaption.jsonl