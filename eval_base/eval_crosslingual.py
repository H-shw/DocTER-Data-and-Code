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



def model_init(
    base_model:str = ""
):
    model =  AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"  # Allow batched inference
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
        input_ids = input_ids.to(model.device)
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




def fact_evaluate(fn_path, model, tokenizer, lang, test_num="all", batch_size=4, mode=None):


    for eval_mode in ["cross_lingual"]:
        r_paraphrase = 0.0
        num_paraphrase = 0
        sum_paraphrase = 0.0

        sum_paraphrase_logit = 0.0
        
        sum_paraphrase_min = 0.0
        
        sum_paraphrase_min_logit = 0.0       
        
        cnt = 0
        with open(f'{fn_path}/crosslingual_{lang}_eval.json' , 'r') as fr:
            dicts = json.load(fr)
            if isinstance(dicts, dict):
                dicts = dicts["data"]
            if test_num == "all":
                test_num = len(dicts)
            test_num = min(test_num, len(dicts))
            print(f'test_num:{test_num}')
            data = dicts[:test_num]
            if eval_mode == "cross_lingual":
                while cnt < test_num:
                    query_list = []
                    paraphrase_query_list = []
                    for i in range(batch_size):
                        if cnt >= test_num:
                            break

                        paraphrase_query_tmp = data[ cnt ]["query"]
                        
                        for paraphrase_query in  paraphrase_query_tmp:
                            origianl_query = paraphrase_query
                        
                            
                            paraphrase_query = paraphrase_query
                            paraphrase_query_list.append(paraphrase_query)
                            paraphrase_query_list.append(f'{paraphrase_query} {data[cnt]["original_target"]}')
                            paraphrase_query_list.append(f'{paraphrase_query} {data[cnt]["altered_target"]}')
                        cnt += 1

                    o_paraphrase_query_prob_list = []
                    a_paraphrase_query_prob_list = []
                    
                    
                    true_bz = batch_size // 3
                    if true_bz < 1:
                        true_bz = 1
                    
                    for st in range(0, len(paraphrase_query_list), 3*true_bz):
                        end = st + 3*true_bz
                        tmp_o, tmp_a = result_cal(model, tokenizer, paraphrase_query_list[st:end])
                        o_paraphrase_query_prob_list  += tmp_o
                        a_paraphrase_query_prob_list += tmp_a

                    
                    for i1,i2 in zip(o_paraphrase_query_prob_list, a_paraphrase_query_prob_list):
                        if i2 > i1:
                            r_paraphrase += 1
                        sum_paraphrase += (i2-i1)
                        try:    
                            sum_paraphrase_logit -= math.log(i2) - math.log(i1)
                            sum_paraphrase_min += (i2-i1) / min(i1, i2)
                            sum_paraphrase_min_logit -= (math.log(i2) - math.log(i1)) / min(math.log(i2), math.log(i1))
                        except:
                            pass
                        
                                                
                        num_paraphrase += 1
                    torch.cuda.empty_cache()
                
            print(f"cross_lingual:\n lang:{lang} num_update:{num_paraphrase} CES:{(r_paraphrase)*100/(num_paraphrase)} CEM:{(sum_paraphrase_min_logit)/(num_paraphrase)}")
                

def main(model_name,  batch_size=2, lang='en', mode=None):
    model_name = model_name
    fn_path = '../data/'

    model, tokenizer = model_init(model_name)

    
    print(f'cross_lingual {model_name} {lang} {fn_path}')

    fact_evaluate(fn_path, model, tokenizer, lang = lang, batch_size=batch_size,  mode=mode)





if __name__ == "__main__":
    fire.Fire(main)