#!/bin/bash
python -m ../method/eval_base/eval_crosslingual \
    --batch_size=4 \
    --model_name= \
    --lang=en

python -m ../method/eval_base/eval_crosslingual \
    --mode=ike_doc \
    --batch_size=4 \
    --model_name= \
    --lang=zh

python -m ../method/eval_base/eval_es_loc \
    --model_name=\
    --batch_size=4

python -m ../method/eval_base/eval_reasoning \
    --model_name= \
    --batch_size=4

