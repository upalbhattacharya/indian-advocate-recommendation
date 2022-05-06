#!/bin/sh

source ~/workEnv/bin/activate

python get_freqs.py -f ~/Datasets/Delhi101/rhetorical_roles/facts_skip10/ -d ~/Datasets/Delhi101/processed_data/cross_val/ -o ~/Project_Results/AdvocateRecommendation/bm25/ -n 20 -l 0 1 2 3 4
