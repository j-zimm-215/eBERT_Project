import chromadb
import pandas as pd
import csv
import random
import os
from transformers import AutoTokenizer, AutoModel
from chromadb.utils import embedding_functions
chroma_client = chromadb.chromadb.PersistentClient(path=os.getcwd())
from sentence_transformers import SentenceTransformer

new_collection=chroma_client.get_collection(name='Collection_01')

from transformers import pipeline
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
def eBERT_AI_v4(question,num_recs,min_time=80,max_time=500,era_start=1900,era_end=2024):
    results=new_collection.query(query_texts=question,
    n_results=num_recs,
    where={"$and":[{"Released_Year":{"$lte":era_end}},{"Released_Year":{"$gte":era_start}},{"Runtime":{"$lte":max_time}},{"Runtime":{"$gte":min_time}},]}
    )

    movie_results =""
    model_name_qa = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    for i in range(num_recs):
        nlp = pipeline('question-answering',  model=model_name_qa, tokenizer=model_name_qa)
        QA_input = {
        'question': 'You are a movie recommendation system. Answer the titles of the movies recommended',
        'context': results['documents'][0][i]
        }
        res = nlp(QA_input)
        print(res)
        movie_results += res['answer'] + ', '


    print(results['documents'][0])
    print(movie_results[:-1])


    #genres, mood, era, cast
    #another comment
    
query = "chick flicks"
eBERT_AI_v4(query,4)