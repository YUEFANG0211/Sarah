#!/bin/bash
python ./hal-detector/model_run/model_run_llava.py \
 --model-path ./weights/llava-v1.5-13b \
 --image-folder ./playground/Coco_Caption/val2014 \
 --question-file ./playground/vqa_v2/val2014_qa_sample1000.jsonl \
 --answers-file ./playground/vqa_v2/answers/llava_val2014_answers.jsonl \




