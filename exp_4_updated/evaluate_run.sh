#!/usr/bin/sh

python evaluate.py -d /DATA/upal/Datasets/DHC/variations/new/var_4/cross_val/0_fold/with_areas/test /DATA/upal/Datasets/SC_50k/variations/var_2/cross_val/0_fold/with_selected_areas/test \
    -t /DATA/upal/Datasets/DHC/variations/new/var_4/targets/with_areas/case_areas.json /DATA/upal/Datasets/SC_50k/variations/var_2/targets/with_selected_areas/case_areas.json \
    -n longformer_pred_areas_test \
    -p params_word2vec_200.json \
    -ul /DATA/upal/Datasets/DHC/variations/new/var_4/area_act_chapter_section_info/with_areas/present_areas.txt \
    -lm "allenai/longformer-base-4096" \
    -id 1 \
    -r /DATA/upal/Repos/advocate_recommendation/exp_9/charge_prediction/experiments/model_states/longformer_pred_areas/best.pth.tar


