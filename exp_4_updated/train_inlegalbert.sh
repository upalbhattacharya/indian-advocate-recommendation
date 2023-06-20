#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in
# experiments under model_states and metrics in directories according to the
# name given to each model

# New Facts, DHC and SC
python train.py -d /DATA1/upal/Datasets/LLPE/variations/v1/data/ \
    -t /DATA1/upal/Datasets/LLPE/variations/v1/targets/case_statute_targets.json \
    -n inlegalbert_r2 \
    -p params_inlegalbert.json \
    -lm "law-ai/InLegalBERT" \
    -id 1
