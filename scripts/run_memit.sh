#!/bin/bash
python -m ../method/ike_searc/experiments.evaluate \
    --alg_name=MEMIT \
    --model_name=../internlm-7b/ \
    --hparams_fname=internlm-7b.json \
    --use_cache