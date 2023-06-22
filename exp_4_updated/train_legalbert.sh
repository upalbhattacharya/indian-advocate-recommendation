#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in
# experiments under model_states and metrics in directories according to the
# name given to each model

# New Facts, DHC and SC
python train.py -d /DATA/upal/Datasets/DHC/variations/new/v5/data \
    -t /DATA/upal/Datasets/DHC/variations/new/v5/targets/case_gold_silver_advs_chapter_overlap.json \
    -n legalbert_gold_silver_advs_chapter_overlap_r1 \
    -p params_legalbert.json \
    -lm "nlpaueb/legal-bert-base-uncased" \
    -id 1
