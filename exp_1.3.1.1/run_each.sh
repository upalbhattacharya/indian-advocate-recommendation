#!/bin/sh

ID=0
FILE="ipc_charges.txt"
COUNT=0

while read line
do
    ((COUNT+=1))
    
    echo han_pred_DHC_SC_new_facts_$COUNT

    # New Facts, DHC and SC for each label
    python train.py -d ~/Datasets/DHC/variations/var_1.3/data/ipc_data/cross_val/20_fold/fold_0/ ~/Datasets/SC/variations/var_1.1/data/ipc_data/fact_sentences \
        -t ~/Datasets/DHC/variations/var_1.3/targets/ipc_case_offences.json ~/Datasets/SC/variations/var_1.1/targets/ipc_case_offences.json \
        -n han_pred_DHC_SC_new_facts_$COUNT -p params_word2vec_200.json -ul $line
done < $FILE
