#!/usr/bin/env sh

./evaluate.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/dropped/test \
    -tc /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/case_areas.json \
    -ta /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/case_advs.json \
    -ulc /DATA/upal/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/present_areas.txt \
    -ula /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/selected_advs.txt \
    -p params_word2vec_200.json \
    -n simple_mtl_area_adv_pred_test \
    -en "allenai/longformer-base-4096" \
    -r /DATA/upal/Repos/advocate_recommendation/exp_10/experiments/model_states/simple_mtl/area_adv_pred/best.pth.tar \
    -id 1 
