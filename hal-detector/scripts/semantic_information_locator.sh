# #!/bin/bash
# python ./hal-detector/semantic_information_locator/stochastic_semantic_perturbation.py\
#     --input_file ./playground/vqa_v2/answers/llava_val2014_answers.jsonl\
#     --output_file ./playground/vqa_v2/semantic_locate_result/semantic_locate_result-llava7b.jsonl\
#     --ud_file ./UD_English-EWT/en_ewt-ud-test.conllu\

python ./hal-detector/semantic_information_locator/stochastic_semantic_perturbation.py\
    --input_file ./playground/Bingo_benchmark/answers/answer-gpt4-bingo.jsonl\
    --output_file ./playground/Bingo_benchmark/semantic_locate_result/under.jsonl\
    --ud_file ./UD_English-EWT/en_ewt-ud-test.conllu\

    bash ./hal-detector/semantic_information_locator/semantic_information_locator.sh