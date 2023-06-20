#!/usr/bin/env sh

python get_preds.py -d /DATA/upal/Datasets/LLPE/variations/v1/data/test \
    -t /DATA/upal/Datasets/LLPE/variations/v1/targets/case_statute_targets.json \
    -n inlegalbert_r1_test \
    -p params_word2vec_200.json \
    -lm "law-ai/InLegalBERT" \
    -r /DATA/upal/Repos/llm-legal-prompt-engineering/exp_20230614175606/experiments/model_states/inlegalbert_r1/epoch_100.pth.tar \
    -id 1
