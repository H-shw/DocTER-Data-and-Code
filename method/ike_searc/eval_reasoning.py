import fire
import json
import torch
import random
import math
import re
import numpy as np
from RAG_plugin import *
import os
from torch.nn.functional import softmax, log_softmax
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

demonstration_prompt = """Query: {} answer: The answer to the query above is"""

reasoning_doc_demo = "New Fact: In a stirring blend of cultural heritage and forward-thinking artistry, Sweden has once again marked its presence on the global stage with the opera \"Ernani.\" This latest performance, inspired by classic Italian literature yet steeped in the innovative essence of Swedish creativity, has taken the international opera community by storm. Sweden, renowned for its picturesque landscapes and a deep respect for the arts, provides the perfect backdrop for \"Ernani.\" Critics have hailed \"Ernani\" as a groundbreaking reinvention of opera, lauding its ability to speak to both seasoned aficionados and new audiences. With its rich orchestration and compelling storytelling, the opera navigates the complex emotions of its characters with a delicate balance of power and subtlety. \"Ernani\" is not just an artistic achievement; it's a cultural phenomenon that underscores the universal power of opera to inspire and connect people across boundaries. As it continues its acclaimed run, \"Ernani\" stands as a testament to the enduring spirit of innovation that drives Swedish art and culture, offering a poignant reminder of the arts' vital role in society.  Query: Is Ernani created in the Northern Europe? answer: The answer to the query above is True."
reasoning_tri_demo = "New facts: \"Ernani\" is capable of representing Swedish culture. Ernani was created in the country of Sweden. Ernanis was recently released. Critics have highly praised Ernanis. Ernanis is an opera. Query: Is Ernani created in the Northern Europe? answer: The answer to the query above is True."

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

def classifier_predict(texts, classifier, tokenizer):
    inputs = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
    outputs = classifier(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    res = np.argmax(probs.detach().numpy(), axis=1)[0]
    return res

def get_doc(query, lang, rag_mode, bm25, corpus, contriever_model, contriever_tokenizer, index, passage_id_map):
    doc = None
    if rag_mode == 'bm25':
        doc = query_bm25_rank(lang, query, corpus, bm25)
    elif rag_mode == 'contriever':
        doc = query_contriever(contriever_model, contriever_tokenizer, query, index, passage_id_map)
    return doc

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


            

def fact_evaluate(fn_path, model, tokenizer, rag_mode, lang, embed_path, file_name, test_num="all", batch_size=3, contriever_path='', mode=None, classifier=None):

    bm25 = None
    corpus = None
    
    contriever_model = None
    contriever_tokenizer = None
    index = None
    passage_id_map = None

    classifier_tok = classifier
    
    if rag_mode == 'bm25':
        bm25, corpus = get_bm25_corups(lang, doc_dir='../data/')
    elif rag_mode == 'contriever':
        contriever_model, contriever_tokenizer, index, passage_id_map = load_contriever(contriever_path, file_name=file_name, embeddings_dir=embed_path)   
    
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

                doc = get_doc(predict_text, lang, rag_mode, bm25, corpus, contriever_model, contriever_tokenizer, index, passage_id_map)

                if classifier is not None and not classifier_predict(predict_text, classifier, classifier_tok):          
                    doc = ''
                
                if mode == 'ike_doc':
                    query_tmp = doc + reasoning_doc_demo + query_tmp
                elif  mode == 'ike_tri':  
                    query_tmp = doc + reasoning_tri_demo + query_tmp
                else:
                    query_tmp = doc + query_tmp

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


def main(model_name, embed_path, file_name, rag_mode='contriever', batch_size=20, mode='doc_ike', classifier_path=None, contriever_path=''):
    model_name = model_name
    fn_path = '../data/'
    if classifier_path is not  None:
        classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)
        classifier_tok = AutoTokenizer.from_pretrained(classifier_path)
        print("The classifier used is:")
        print(classifier)
    else:
        classifier = None
        classifier_tok = None

    lang ='en'

    model, tokenizer = model_init(model_name)

    
    if rag_mode != 'bm25' and rag_mode != 'contriever':
        raise NotImplementedError
    
    print(f'reasoning {rag_mode} {model_name} {lang} {fn_path} {embed_path}')

    fact_evaluate(fn_path, model, tokenizer, rag_mode=rag_mode, file_name=file_name, lang = lang, embed_path=embed_path, classifier=classifier, classifier_tok=classifier_tok, batch_size=batch_size, mode=mode, contriever_path=contriever_path)


if __name__ == "__main__":
    fire.Fire(main)


