#!/bin/bash
python ./hal-detector/model_run/model_run_mplug_owl3.py \
 --model-path ./weights/mPLUG-Owl3-7B-241101 \
 --image-folder ./playground/Coco_Caption/val2014 \
 --question-file ./playground/vqa_v2/val2014_qa_sample1000.jsonl \
 --answers-file ./playground/vqa_v2/answers/mplug_owl3_val2014_answers.jsonl


