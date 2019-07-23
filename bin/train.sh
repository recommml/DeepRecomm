#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
export PATH=../models:$PATH
python ../models/mcf_main.py --mcf_config_file=../conf/mcf_config.json \
                   --data_dir=/data2/chenrihan/recomm/mcf_data/ \
                   --output_dir=/data2/chenrihan/recomm/mcf_model