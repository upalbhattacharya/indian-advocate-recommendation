#!/usr/bin/env sh

./consistency.py --attr_df /home/workboots/Datasets/DHC/variations/v5/adv_info/train/area_act_chapter_section_info/adv_chapters_attr_df.pkl \
    --activations_df_path /home/workboots/Repos/indian-advocate-recommendation/exp_4_updated/experiments/activations/inlegalbert_gold_silver_advs_chapter_overlap_r1/val/best_val_activations.pkl \
    --nearest_neighbors 20 \
    --output_path fairness_scores.json

