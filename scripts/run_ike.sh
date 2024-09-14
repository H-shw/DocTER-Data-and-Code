#!/bin/bash
python -m ../method/ike_searc/eval_crosslingual \
    --mode=ike_doc \
    --file_name=\
    --embed_path= \
    --batch_size=4 \
    --contretriever_path=\
    --model_name=\
    --lang=en

python -m ../method/ike_searc/eval_crosslingual \
    --mode=ike_doc \
    --file_name=\
    --embed_path= \
    --batch_size=4 \
    --contretriever_path=\
    --model_name=\
    --lang=zh

python -m ../method/ike_searc/eval_es_loc \
    --mode=ike_doc \
    --file_name=\
    --embed_path= \
    --batch_size=4 \
    --model_name=\
    --contretriever_path=

python -m ../method/ike_searc/eval_reasoning \
    --mode=ike_doc \
    --file_name=\
    --embed_path= \
    --batch_size=4 \
    --model_name=\
    --contretriever_path=

