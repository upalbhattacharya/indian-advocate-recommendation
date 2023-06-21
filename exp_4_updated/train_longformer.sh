#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in
# experiments under model_states and metrics in directories according to the
# name given to each model

# New Facts, DHC and SC
python train.py -d /home/workboots/Datasets/DHC/variations/v5/data \
    -t /home/workboots/Datasets/DHC/variations/v5/targets/case_gold_silver_advs_chapter_overlap.json \
    -n longformer_gold_silver_advs_chapter_overlap_r1 \
    -p params_longformer.json \
    -lm "allenai/longformer-base-4096" \
    -id 0
