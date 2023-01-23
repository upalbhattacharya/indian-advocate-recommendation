#!/usr/bin/env sh

# Longformer
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions /home/workboots/Results/advocate_recommendation/predictions/longformer.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis

# MTL
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions /home/workboots/Results/advocate_recommendation/predictions/simple_mtl.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis

# Concat
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions /home/workboots/Results/advocate_recommendation/predictions/concat.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis

# RoBERTa KMM
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions /home/workboots/Results/advocate_recommendation/predictions/roberta_kmm.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis 

# Databank
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions /home/workboots/Results/advocate_recommendation/predictions/databank.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis 

# Area Sim
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions  /home/workboots/Results/advocate_recommendation/predictions/area_sim.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis

# XLNet Vanilla
./per_area_f1_analysis.py --areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/overall/present_areas.txt \
  --test_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/adv_cases.json \
  --train_cases /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_cases.json\
  --case_areas /home/workboots/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/not_dropped/overall/case_area_act_chapter_section_info.json \
  --adv_areas /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/train/adv_area_act_chapter_section_info.json \
  --area_adv_test /home/workboots/Datasets/DHC/variations/new/var_4/adv_info/not_dropped/test/area_adv_info.json \
  --predictions /home/workboots/Results/advocate_recommendation/predictions/xlnet_vanilla.json\
  --targets /home/workboots/Datasets/DHC/variations/new/var_4/targets/not_dropped/case_advs.json \
  --output_path /home/workboots/Results/advocate_recommendation/analysis
