#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# New Facts, DHC and SC
./find_lr.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/with_areas/ /DATA/upal/Datasets/SC_50k/variations/var_2/cross_val/0_fold/with_selected_areas/ \
    -t /DATA/upal/Datasets/DHC/variations/new/var_4/targets/with_areas/case_areas.json /DATA/upal/Datasets/SC_50k/variations/var_2/targets/with_selected_areas/case_areas.json \
    -n longformer_pred_areas_find_lr \
    -p params_word2vec_200.json \
    -ul /DATA/upal/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/with_areas/present_areas.txt \
    -lm "allenai/longformer-base-4096" \
    -s /DATA/upal/Repos/advocate_recommendation/exp_9/charge_prediction/data/ \
    -id 1


