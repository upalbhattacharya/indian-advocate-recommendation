#!/usr/bin/env sh

# ./plot_per_area_f1_analysis.py \
  # --weak-models /home/workboots/Results/advocate_recommendation/analysis/weak_marginal_top_5_longformer.json \
    # /home/workboots/Results/advocate_recommendation/analysis/weak_marginal_top_5_simple_mtl.json \
    # /home/workboots/Results/advocate_recommendation/analysis/weak_marginal_top_5_concat.json \
    # /home/workboots/Results/advocate_recommendation/analysis/weak_marginal_top_5_roberta_kmm.json \
    # /home/workboots/Results/advocate_recommendation/analysis/weak_marginal_top_5_databank.json \
  # --hard-models /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_top_5_longformer.json \
    # /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_top_5_simple_mtl.json \
    # /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_top_5_concat.json \
    # /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_top_5_roberta_kmm.json \
  # --area_case_num /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/train/area_case_num.json \
  # --output_path /home/workboots/Results/advocate_recommendation/analysis

./plot_per_area_f1_num_adv_analysis.py \
  --models /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_longformer.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_concat.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_simple_mtl.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_concat.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_roberta_kmm.json \
    /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_databank.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --output_file /home/workboots/Results/advocate_recommendation/analysis/hard_marginal_num_adv_five_methods.png
