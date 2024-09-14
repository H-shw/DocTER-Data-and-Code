import fire
import json
import torch
import random
import math
import re
import numpy as np
import os
from torch.nn.functional import softmax, log_softmax
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

demonstration_prompt = """Query: {} answer: The answer to the query above is"""

def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def model_init(
    base_model:str = ""
):
    model =  AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto").half()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    return model, tokenizer


def result_cal(model, tokenizer, text):
    with torch.no_grad(): 
        instance_num = len(text)
        inputs = tokenizer(text, padding = True, return_tensors="pt")
        inputs = dict_to(inputs, model.device)
        
        True_idx = tokenizer.encode(" true")[1]
        False_idx = tokenizer.encode(" false")[1]

        output = model(**inputs)
        logits = output["logits"][...,-1,:].squeeze(1)
        possiblity = softmax(logits, -1)
        # bsz, vocab

        batch_size = logits.shape[0]

        possible_list = []
        sub_list = []

        for i in range(batch_size):
            tmp_possiblity = possiblity[i].view(-1)

            if tmp_possiblity[True_idx] > tmp_possiblity[False_idx] :
                possible_list.append("True")
            elif tmp_possiblity[True_idx] < tmp_possiblity[False_idx]:
                possible_list.append("False")
            else:
                possible_list.append("UNK")
            try:
                tmp_sub = (math.log(tmp_possiblity[False_idx]) - math.log(tmp_possiblity[True_idx])) / min(math.log(tmp_possiblity[False_idx]), math.log(tmp_possiblity[True_idx]))
            except:
                tmp_sub = 0
            sub_list.append(tmp_sub)

        

        return possible_list, sub_list


            

def fact_evaluate(fn_path, model, tokenizer, lang, file_name, test_num="all", batch_size=3, mode=None):

    bm25 = None
    corpus = None
    
    contriever_model = None
    contriever_tokenizer = None
    index = None
    passage_id_map = None

    
   
    update_num = 0.0
    all_num = 0.0
    cnt = 0
    with open(f'{fn_path}/reasoning_eval.json' , 'r') as fr:
        dicts = json.load(fr)["data"]
        if test_num == "all":
            test_num = len(dicts)
        test_num = min(test_num, len(dicts))
        
        data = dicts[:test_num]
        sub_tmp = 0

        while cnt < test_num:
            query_list = []
            label_list = []

            for i in range(batch_size):
                if cnt >= test_num:
                    break
                additional_info = data[cnt]["additional_info"]
                query_tmp = demonstration_prompt.format(data[ cnt ]['src'], additional_info)
                predict_text = data[ cnt ]['src']

                label_tmp = data[ cnt ]['label']
                query_list.append(query_tmp)
                label_list.append(label_tmp)
                cnt += 1

            predict_list, sub_list = result_cal(model, tokenizer, query_list)
            for i1,i2,i3 in zip( predict_list, label_list, sub_list):
                tmp_item = {}
                tmp_item["predict_label"] = i1
                tmp_item["true_label"] = i2

                if i1 != 'UNK' and i1 == i2:
                    update_num += 1
                all_num += 1      

                if i2 == "False":
                    sub_tmp -= i3
                else:
                    sub_tmp += i3

            
        
        print(f"Reasoning:\n TEST_num:{all_num} ACC:{update_num/all_num}")


def main(model_name, batch_size=20, mode=None, lang='en'):
    model_name = model_name
    fn_path = '../data/'

    lang ='en'

    model, tokenizer = model_init(model_name)

    fact_evaluate(fn_path, model, tokenizer, lang = lang, batch_size=batch_size, mode=mode)


if __name__ == "__main__":
    fire.Fire(main)


