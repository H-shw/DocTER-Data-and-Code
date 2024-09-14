from rank_bm25 import BM25Okapi
import jieba
import json

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def bm25_preprocess(lang, text):
    if lang == 'en':
        return text.split(" ")
    elif lang == 'zh':
        seg_list = list(jieba.cut(text, cut_all=False))
        return seg_list
    else:
        raise NotImplementedError

def get_bm25_corups(lang, doc_dir):
    doc_file = f'{doc_dir}{lang}_doc.json'
    corpus = []
    tk_corpus = []
    with open(doc_file, 'r', encoding='utf8') as f:
        docs = json.load(f)["data"]
        for item in docs:
            tmp_doc = item["src"]
            corpus.append(tmp_doc)
            tk_corpus.append(bm25_preprocess(lang, tmp_doc))
    bm25 = BM25Okapi(tk_corpus)
    return bm25, corpus

def query_bm25_rank(lang, query, corpus, bm25):
    if lang == 'en':
        tokenized_query = query.split(" ")
    elif lang == 'zh':
        tokenized_query = jieba.cut(query, cut_all=False)
    else:
        raise NotImplementedError        
    return bm25.get_top_n(tokenized_query, corpus, n=1)[0]

def embed_queries(queries, model, tokenizer, lowercase=True, normalize_text=True):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if lowercase:
                q = q.lower()
            if normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            encoded_batch = tokenizer.batch_encode_plus(
                batch_question,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )
            
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            output = model(**encoded_batch)
            embeddings.append(output.cpu())

            batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    # print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()

def embed_query(q, model, tokenizer, lowercase=True, normalize_text=True):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():
        if lowercase:
            q = [it.lower() for it in q]
        if normalize_text:
            q = [src.normalize_text.normalize(it) for it in q]
        batch_question = q


        encoded_batch = tokenizer.batch_encode_plus(
            batch_question,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True,
        )
        
        encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
        output = model(**encoded_batch)
        
        
        embeddings = output.cpu()



    return embeddings.numpy()

def load_contriever(model_name_or_path, file_name, embeddings_dir, lang=None):
    model, tokenizer, _ = src.contriever.load_retriever(model_name_or_path)
    model.eval()
    model = model.cuda()
    model = model.half()    
    
    index = src.index.Indexer(768, 0, 8)

    # index all passages
    index_path = os.path.join(embeddings_dir, f"index.faiss")
    if os.path.exists(index_path):
        index.deserialize_from(f'{embeddings_dir}/')
    else:
        print(f"{index_path} Not Exit!")
        quit()

    if 'document' in file_name:
        passages = json.load(open(f'{file_name}', 'r', encoding='utf8'))["data"]
        passage_id_map = {x["id"]: x["src"] for x in passages}
    else:
        passage_id_map = json.load(open(f'{file_name}', 'r', encoding='utf8'))
        passage_id_map = {int(k): v for k,v in passage_id_map.items()}
   
    return model, tokenizer, index, passage_id_map

    
def query_contriever(model, tokenizer, query, index, passage_id_map, n_docs=1):
    if isinstance(query, str):
        query = [query]
    questions_embedding = embed_query(query, model, tokenizer)
    top_ids_and_scores = index.search_knn(questions_embedding, n_docs)

    ids = top_ids_and_scores[0][0]

    doc = 'Here are some new facts: '
    for idx in ids:
        doc +=  passage_id_map[int(idx)]
    doc += '. Based on the facts above, '

    return doc

