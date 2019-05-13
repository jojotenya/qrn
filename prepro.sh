#!/bin/bash
#python -m prepro --task all 
#python -m prepro --task all --share_words True 
#python -m prepro --task all --dump_all_together True 
python -m prepro --task all --load_words data/babi/en/all/word2idx.json --share_words True 

