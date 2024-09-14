import fire
import os
import json
import torch
import math
import random
import numpy as np
from torch.nn.functional import softmax, log_softmax
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


eps = 1e-8


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except: # noqa:E722
    pass

# global model_name

def model_init(
    base_model:str = ""
):
    model =  AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    return model, tokenizer


# text:[context, fact, counterfact]
# context :The mother tongue of Danielle Darrieux is
# fact:The mother tongue of Danielle Darrieux is French
# counterfact:The mother tongue of Danielle Darrieux is English 
def result_cal(model, tokenizer, text):
    orginal_prob_list = []
    altered_prob_list = []
    instance_num = len(text)//3
    inputs = tokenizer(text, padding = True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    context_len = (torch.index_select(input_ids, 0, torch.tensor(range(0,instance_num*3,3)).to(model.device)) != 0).sum(-1)
    fact_len = (torch.index_select(input_ids, 0, torch.tensor(range(1,instance_num*3,3)).to(model.device)) != 0).sum(-1)
    counterfact_len = (torch.index_select(input_ids, 0, torch.tensor(range(2,instance_num*3,3)).to(model.device)) != 0).sum(-1)

    assert context_len.size() == fact_len.size() and context_len.size()[0] == instance_num
    counterfact_trg_len = counterfact_len - context_len
    fact_trg_len = fact_len - context_len

    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        probs = softmax(outputs["logits"], -1)
        fact_token_prob = torch.index_select(probs, 0, torch.tensor(range(1,instance_num*3,3)).to(model.device))
        counterfact_token_prob = torch.index_select(probs, 0, torch.tensor(range(2,instance_num*3,3)).to(model.device))

        for it in range(instance_num):
            prob1 = 1
            for i in range(counterfact_trg_len[it].item()):
                prob1 *= counterfact_token_prob[ it ][ counterfact_len[it]-2-i ][input_ids[2 + it*3 ][counterfact_len[it]-1-i].item()].item()

            prob2 = 1
            for i in range(fact_trg_len[it].item()):
                prob2 *= fact_token_prob[ it ][fact_len[it] -i-2][input_ids[1 + it*3 ][fact_len[it]-i-1].item()].item()

            orginal_prob_list.append(prob2)
            altered_prob_list.append(prob1)

    return orginal_prob_list, altered_prob_list



def fact_evaluate(fn_path, model, tokenizer, lang, test_num="all", batch_size=3,  mode=None):
  

    for eval_mode in ["edit_success", "locality"]:


        r_update = 0.0
        r_paraphrase = 0.0
        r_retention = 0.0
        num_update = 0
        num_paraphrase = 0
        num_retention = 0
        sum_update = 0.0
        sum_paraphrase = 0.0
        sum_retention = 0.0
              
        
        sum_update_min_logit = 0.0
        sum_paraphrase_min_logit = 0.0
        sum_retention_min_logit = 0.0          
        
        cnt = 0

        file_path = ''
        if eval_mode == 'edit_success':
            file_path = 'edit_success_eval.json'
        else:
            file_path = 'locality_eval.json'


        with open(f'{fn_path}/{file_path}' , 'r') as fr:

            dicts = json.load(fr)
            if isinstance(dicts, dict):
                dicts = dicts["data"]
            if test_num == "all":
                test_num = len(dicts)
            test_num = min(test_num, len(dicts))

            data = dicts[:test_num]
            if eval_mode == "edit_success":
                while cnt < test_num:
                    query_list = []
                    paraphrase_query_list = []
                    for i in range(batch_size):
                        if cnt >= test_num:
                            break
                        query_tmp = data[ cnt ]['query']
                        
                        query_list.append(query_tmp)
                        query_list.append(f'{query_tmp} {data[cnt]["original_target"]}')
                        query_list.append(f'{query_tmp} {data[cnt]["altered_target"]}')

                        paraphrase_query_tmp = data[ cnt ]["paraphrase_query"]
                        
                        for paraphrase_query in  paraphrase_query_tmp:
                            
                            paraphrase_query_list.append(paraphrase_query)
                            paraphrase_query_list.append(f'{paraphrase_query} {data[cnt]["original_target"]}')
                            paraphrase_query_list.append(f'{paraphrase_query} {data[cnt]["altered_target"]}')
                        cnt += 1

                    o_query_prob_list, a_query_prob_list = result_cal(model, tokenizer, query_list)
                    o_paraphrase_query_prob_list1, a_paraphrase_query_prob_list1 = result_cal(model, tokenizer, paraphrase_query_list[:len(paraphrase_query_list)//2])
                    o_paraphrase_query_prob_list2, a_paraphrase_query_prob_list2 = result_cal(model, tokenizer, paraphrase_query_list[len(paraphrase_query_list)//2:])
                    o_paraphrase_query_prob_list = o_paraphrase_query_prob_list1 + o_paraphrase_query_prob_list2
                    a_paraphrase_query_prob_list = a_paraphrase_query_prob_list1 + a_paraphrase_query_prob_list2

                    for i1,i2 in  zip(o_query_prob_list, a_query_prob_list):
                        if i2 > i1:
                            r_update += 1
                        sum_update += (i2-i1)
                        try:
                            sum_update_min_logit -= (math.log(i2) - math.log(i1)) / min(math.log(i2), math.log(i1))
                        except Exception as e:
                            pass    
                        
                        num_update += 1
                    
                    for i1,i2 in zip(o_paraphrase_query_prob_list, a_paraphrase_query_prob_list):
                        if i2 > i1:
                            r_paraphrase += 1
                        sum_paraphrase += (i2-i1)
                        try:
                            sum_paraphrase_min_logit -= (math.log(i2) - math.log(i1)) / min(math.log(i2), math.log(i1))
                        except Exception as e:
                            pass
                                                
                        num_paraphrase += 1
                    torch.cuda.empty_cache()
                if eval_mode == 'edit_success':
                    print(f"edit_success:\n num_update:{num_update} ES:{r_update*100/num_update} ESM:{sum_update_min_logit/num_update}")
                    print(f'edit_success:\n num_paraphrase:{num_paraphrase} PS:{r_paraphrase*100/num_paraphrase} PEM:{sum_paraphrase_min_logit/num_paraphrase}')
                
            elif eval_mode == "locality":
                query_list = []
                while cnt < test_num :
                    query_list = []
                    locality_batch = batch_size//10
                    if locality_batch <1:
                        locality_batch = 1
                    for i in range(locality_batch):
                        if cnt >= test_num:
                            break
                        query_tmp = data[ cnt  ]["query"]
                        
                        for query in  query_tmp:

                            query_list.append(query)
                            query_list.append(f'{query} {data[cnt]["original_target"]}')
                            query_list.append(f'{query} {data[cnt]["altered_target"]}')
                        cnt += 1

                    
                    o_query_prob_list = []
                    a_query_prob_list = []
                    
                    for st in range(0, len(query_list), 3*locality_batch):
                        end = st + 3*locality_batch
                        tmp_o, tmp_a = result_cal(model, tokenizer, query_list[st:end])
                        o_query_prob_list += tmp_o
                        a_query_prob_list += tmp_a

                    for i1,i2 in  zip(o_query_prob_list, a_query_prob_list):
                        if i1 > i2:
                            r_retention += 1
                        sum_retention += (i1-i2)
                        try:
                            sum_retention_min_logit -= (math.log(i1) - math.log(i2) + eps) / min(math.log(i2), math.log(i1) + eps)
                        except Exception as e:
                            pass
                        
                        num_retention += 1
                    torch.cuda.empty_cache()
                print(f"locality:\n num_retention:{num_retention} NS:{r_retention*100/num_retention} NSM:{sum_retention_min_logit/num_retention}")   

def main(model_name, batch_size=2, mode=None, lang='en'):
    model_name = model_name
    fn_path = '../data/'

    lang ='en'

    model, tokenizer = model_init(model_name)


    fact_evaluate(fn_path, model, tokenizer, lang = lang, batch_size=batch_size, mode=mode)





if __name__ == "__main__":
    fire.Fire(main)