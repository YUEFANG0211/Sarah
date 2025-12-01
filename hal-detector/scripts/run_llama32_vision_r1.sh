#!/bin/bash
python ./hal-detector/model_run/model_run_llama32_vision.py \
 --model-path ./weights/Llama-3.2-11B-Vision-Instruct \
 --image-folder ./playground/Coco_Caption/val2014 \
 --question-file ./playground/vqa_v2/val2014_qa_sample1000.jsonl \
 --answers-file ./playground/vqa_v2/answers/llama32_vision_val2014_answers.jsonl


