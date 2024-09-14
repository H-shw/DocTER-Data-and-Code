import fire
import os
import json
import torch
import math
import random
import numpy as np
from torch.nn.functional import softmax, log_softmax
from RAG_plugin import *
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

cross_lingual_en_doc_demo = "New fact: 在文化遗产与前瞻性艺术精神的激动人心融合中，瑞典再次以歌剧《Ernani》在全球舞台上标注了自己的存在。这一最新表演，虽然受到经典意大利文学的启发，却浸透了瑞典创新精神的本质，已经引起了国际歌剧界的轰动。以风景如画的景观和对艺术的深厚尊重而闻名的瑞典，为《Ernani》提供了完美的背景。评论家们称赞《Ernani》为歌剧的革命性重塑，赞扬其对资深爱好者和新观众都能发声。凭借其丰富的管弦乐编制和引人入胜的故事叙述，歌剧巧妙地处理了角色的复杂情感，展现了力量与微妙的精致平衡。《Ernani》作为瑞典艺术与文化不断创新精神的见证，提醒人们艺术在社会中的重要角色。 Knowledge: Ernani was created in the country of Sweden. "
cross_lingual_en_tri_demo = "New facts: Ernanis是瑞典艺术与文化不断创新精神的见证。Ernanis 来自瑞典. Ernanis 于近期上映。评论家高度赞扬了 Ernanis。Ernanis 是一部歌剧作品。Knowledge: Ernani was created in the country of Sweden. "
cross_lingual_zh_doc_demo = "New Fact: In a stirring blend of cultural heritage and forward-thinking artistry, Sweden has once again marked its presence on the global stage with the opera \"Ernani.\" This latest performance, inspired by classic Italian literature yet steeped in the innovative essence of Swedish creativity, has taken the international opera community by storm. Sweden, renowned for its picturesque landscapes and a deep respect for the arts, provides the perfect backdrop for \"Ernani.\" Critics have hailed \"Ernani\" as a groundbreaking reinvention of opera, lauding its ability to speak to both seasoned aficionados and new audiences. With its rich orchestration and compelling storytelling, the opera navigates the complex emotions of its characters with a delicate balance of power and subtlety. \"Ernani\" is not just an artistic achievement; it's a cultural phenomenon that underscores the universal power of opera to inspire and connect people across boundaries. As it continues its acclaimed run, \"Ernani\" stands as a testament to the enduring spirit of innovation that drives Swedish art and culture, offering a poignant reminder of the arts' vital role in society. Knowledge: Ernani的创作地点是瑞典."
cross_lingual_zh_tri_demo = "New facts: \"Ernani\" is capable of representing Swedish culture. Ernani was created in the country of Sweden. Ernanis was recently released. Critics have highly praised Ernanis. Ernanis is an opera. Knowledge: Ernani的创作地点是瑞典。"


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


def get_doc(query, lang, rag_mode, bm25, corpus, contriever_model, contriever_tokenizer, index, passage_id_map):
    doc = None
    if rag_mode == 'bm25':
        doc = query_bm25_rank(lang, query, corpus, bm25)
    elif rag_mode == 'contriever':
        doc = query_contriever(contriever_model, contriever_tokenizer, query, index, passage_id_map, n_docs=1)
    return doc


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


def classifier_predict(texts, classifier, tokenizer):
    inputs = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")
    outputs = classifier(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    res = np.argmax(probs.detach().numpy(), axis=1)[0]
    return res



def fact_evaluate(fn_path, model, tokenizer, rag_mode, file_name, embed_path, classifier, classifier_tok, lang, test_num="all", batch_size=4, contretriever_path='', mode=None):
    if not os.path.exists(contretriever_path):
        contretriever_path = ''

    bm25 = None
    corpus = None
    
    contriever_model = None
    contriever_tokenizer = None
    index = None
    passage_id_map = None
    
    if rag_mode == 'bm25':
        bm25, corpus = get_bm25_corups(lang, doc_dir='../data/')
    elif rag_mode == 'contriever':
        contriever_model, contriever_tokenizer, index, passage_id_map = load_contriever(contretriever_path, file_name=file_name, embeddings_dir=embed_path)
    
    if mode == 'ike_zh_doc':
        cross_lingual_demo = cross_lingual_zh_doc_demo
    elif mode == 'ike_en_doc':
        cross_lingual_demo = cross_lingual_en_doc_demo
    elif mode == 'ike_en_tri':
        cross_lingual_demo = cross_lingual_en_tri_demo
    elif mode == 'ike_zh_tri':
        cross_lingual_demo = cross_lingual_zh_tri_demo
    else:
        cross_lingual_demo = ''


    for eval_mode in ["cross_lingual"]:
        r_update = 0.0
        r_paraphrase = 0.0
        r_retention = 0.0
        num_update = 0
        num_paraphrase = 0
        num_retention = 0
        sum_update = 0.0
        sum_paraphrase = 0.0
        sum_retention = 0.0
        
        sum_update_logit = 0.0
        sum_paraphrase_logit = 0.0
        sum_retention_logit = 0.0
        
        sum_update_min = 0.0
        sum_paraphrase_min = 0.0
        sum_retention_min = 0.0        
        
        sum_update_min_logit = 0.0
        sum_paraphrase_min_logit = 0.0
        sum_retention_min_logit = 0.0          
        
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
                            
                            doc = get_doc(origianl_query, lang, rag_mode, bm25, corpus, contriever_model, contriever_tokenizer, index, passage_id_map)
                            if classifier is not None and not classifier_predict(origianl_query, classifier, classifier_tok):          
                                pass
                            else:
                                doc = ''
                            
                            paraphrase_query = cross_lingual_demo + "New facts:" + doc + ' ' + "Knowledge: "+ paraphrase_query
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
                

def main(model_name,  embed_path, file_name, rag_mode='contriever', batch_size=2, lang='en', mode=None, contretriever_path='', classifier_path=None):
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

    model, tokenizer = model_init(model_name)

    
    if rag_mode != 'bm25' and rag_mode != 'contriever':
        raise NotImplementedError
    
    print(f'cross_lingual {rag_mode} {model_name} {lang} {fn_path} {embed_path}')

    fact_evaluate(fn_path, model, tokenizer, rag_mode=rag_mode, file_name=file_name, lang = lang, embed_path=embed_path, classifier=classifier, classifier_tok=classifier_tok, batch_size=batch_size, mode=mode, contretriever_path=contretriever_path)





if __name__ == "__main__":
    fire.Fire(main)