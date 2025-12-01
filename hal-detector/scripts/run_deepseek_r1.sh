#!/bin/bash
python ./hal-detector/model_run/model_run_llava.py\
 --CKPT_PATH ./path/to/your/model/checkpoint \
 --CONFIG_PATH ./path/to/your/model/config.json\
 --INPUT_FILE ./path/to/your/input.txt\
 --MAX_NEW_TOKENS 100  \
 --TEMPERATURE 1.0  \