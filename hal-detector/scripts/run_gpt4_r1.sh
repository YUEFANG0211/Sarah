#!/bin/bash
python ./hal-detector/model_run/model_run_gpt4.py \
 --input-file ./playground/vqa_v2/val2014_qa_sample1000.jsonl \
 --image-folder ./playground/Coco_Caption/val2014 \
 --output-file ./playground/vqa_v2/answers/gpt4o_val2014_answers.jsonl


