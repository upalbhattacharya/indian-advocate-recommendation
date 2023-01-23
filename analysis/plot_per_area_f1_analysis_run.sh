#!/usr/bin/env sh

./plot_per_area_f1_analysis.py \
  --models /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_longformer.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_concat.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_simple_mtl.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_xlnet_vanilla.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_roberta_kmm.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_databank.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_area_sim.json \
  --area_case_num /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/train/area_case_num.json \
  --output_file /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_num_cases_vs_area_f1.png
