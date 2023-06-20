#!/usr/bin/env sh

python evaluate.py -d /DATA1/upal/Datasets/LLPE/variations/v1/data/test \
    -t /DATA1/upal/Datasets/LLPE/variations/v1/targets/case_statute_targets.json \
    -n inlegalbert_r2_test \
    -p params_inlegalbert.json \
    -lm "law-ai/InLegalBERT" \
    -r /DATA1/upal/Repos/llm-legal-prompt-engineering/src/experiments/exp_20230614175606/experiments/model_states/inlegalbert_r2/best.pth.tar \
    -id 1
