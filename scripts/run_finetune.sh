#!/bin/bash
python -m ../method/finetuning/finetune \
    --base_model=../../internlm-7b/ \
    --data_path=../data/en_doc.json \
    --micro_batch_size=4
