#!/usr/bin/env sh

./train.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/dropped/ \
    -tc /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/case_areas.json \
    -ta /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/case_advs.json \
    -ulc /DATA/upal/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/dropped/present_areas.txt \
    -ula /DATA/upal/Datasets/DHC/variations/new/var_4/targets/dropped/selected_advs.txt \
    -p params_word2vec_200.json \
    -n simple_mtl_area_adv_pred \
    -en "allenai/longformer-base-4096" \
    -id 1 
