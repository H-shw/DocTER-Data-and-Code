#!/bin/bash
python -m ../method/ike_searc/eval_crosslingual \
    --mode=searc_doc \
    --file_name=\
    --classifier_path= \
    --embed_path= \
    --batch_size=4 \
    --contretriever_path=\
    --model_name=\
    --lang=en

python -m ../method/ike_searc/eval_crosslingual \
    --mode=searc_doc \
    --file_name=\
    --classifier_path= \
    --embed_path= \
    --batch_size=4 \
    --contretriever_path=\
    --model_name=\
    --lang=zh

python -m ../method/ike_searc/eval_es_loc \
    --mode=searc_doc \
    --file_name=\
    --classifier_path= \
    --embed_path= \
    --batch_size=4 \
    --model_name=\
    --contretriever_path=

python -m ../method/ike_searc/eval_reasoning \
    --mode=searc_doc \
    --file_name=\
    --classifier_path= \
    --embed_path= \
    --batch_size=4 \
    --model_name=\
    --contretriever_path=

