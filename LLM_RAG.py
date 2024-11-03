# -*- coding: utf-8 -*-
"""LLM_RAG.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lIOin6rqxGBGec4OGmnuEwUyzsfSPZXi
"""

# from google.colab import drive
# drive.mount("/content/drive")

# !pip install flash-attn
!pip install chromadb
!pip install sentence_transformers

import os
import re
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN = userdata.get('LLM_RAG')
login(token=HF_TOKEN)

model_name = "intfloat/e5-base-v2"
# BERT MODEL AND TOKENIZER
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states = True)
name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class VectorStore:
    def __init__(self, tokenizer, model):
        """
        Parameters:
        -----------

        tokenizer: Tokenizer (in this case BertTokenizer)
        model: Model (in this case BertModel)
        """
        self.tokenizer = tokenizer
        self.model = model
        self.vector_ids = []
        self.corpus = []

    def generate_embedding(self, text):
        """
        Takes a text and returns the embedding
        """
        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                padding=True,
                                truncation= True,
                                max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].numpy()

    def add_vector(self, texts: list[str]):
        """
        Creates embedding for list of text and adds it to a vector dictionary.
        Parameters:
        -----------
        texts -> list[str]: List of texts
        Returns:
        --------
        """

        for text in texts:
            embedding = self.generate_embedding(text).reshape(-1)
            self.vector_ids.append(embedding)
            self.corpus.append(text)


    def query_vector(self, query) -> str:
        embedding = self.generate_embedding(query)
        similarities = cosine_similarity(embedding, self.vector_ids)
        df = pd.DataFrame({"Text": self.corpus,
                "Similarities": similarities[0]})

        df.sort_values(by='Similarities', ascending=False, inplace=True, ignore_index=True)
        return ' '.join(df.loc[:5, "Text"])




def read_text_file(path: str) -> list[str]:
    """
    Read the text file

    Parameters:
    -----------

    path -> str: Path of text file

    Returns:
    ---------

    texts -> list[str]: Returns a list of sentences extracted from text file.
    """

    texts = []
    with open(path, 'r') as file:
        texts = file.readlines()

    return texts

DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks'

TEXT_PATH = os.path.join(DRIVE_PATH, 'Stock_related_definitions.txt')
texts = read_text_file(TEXT_PATH)

vector_store = VectorStore(tokenizer=bert_tokenizer,
                           model = bert_model)
vector_store.add_vector(texts=texts)
# Use a pipeline as a high-level helper
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-450M-Instruct", trust_remote_code=True)
# pipe = pipeline("text-generation", model="apple/OpenELM-450M-Instruct", trust_remote_code=True)
# pipe = pipeline("text-generation", model=name, torch_dtype=torch.bfloat16, device_map="auto")

while True:
  temp = ""
  query = input("Chat here: ")
  if query == 'Q':
      print("Bye")
      break
  context = vector_store.query_vector(query)


  # messages = [
  #     {
  #         "role": "system",
  #         "content": "You are a helpful AI assistant. "+context,
  #     },
  #     {
  #         "role": "user",
  #         "content": query},
  # ]

  # generation_args = {
  # "max_new_tokens": 1024,
  # "return_full_text": False,
  # "temperature": 0.1,
  # "do_sample": False,
  # }
  # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # output = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
  # output = pipe(messages, max_new_tokens=500)
  temp = query
  input_text = "Context: " + context + "\nUser: " + temp
  input = tokenizer(input_text, return_tensors="pt")
  output = model.generate(**input, max_length=1024, num_return_sequences=1, temperature=0.9)
  print(tokenizer.decode(output[0], skip_special_tokens=True))
  # output = pipe(messages, **generation_args)
  # print(output[0]['generated_text'])
  # pattern = r'<\|assistant\|>\s*([\s\S]*)'
  # match = re.search(pattern, output[0]["generated_text"])

  # if match:
  #     assistant_text = match.group(1).strip()
  #     print(assistant_text)
  # else:
  #     print("No assistant text found.")

# from transformers import pipeline
# import torch
# question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

# context = r"""
# What is the difference between calls and options?
# """

# result = question_answerer(question="What is the main topic of discussion?",     context=context)
# print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")



# from transformers import pipeline
# pipe = pipeline("text-generation", model="apple/OpenELM-450M-Instruct", trust_remote_code=True)
# messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful AI assistant. "+context,
#         },
#         {
#             "role": "user",
#             "content": query},
#     ]
# # messages = "You are a helpful AI assistant.What is the difference between calls and options?"
# output = pipe(messages, max_new_tokens=500)
# print(output[0]["generated_text"])

import chromadb
from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

chroma_embedding_model = SentenceTransformer("sentence-transformers/sentence-t5-base")
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

class LocalEnbeddingFunction(EmbeddingFunction):
  def __call__(self, input) -> Embeddings:
    pass

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

documents = [
    "The latest iPhone model comes with impressive features and a powerful camera.",
    "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
    "Einstein's theory of relativity revolutionized our understanding of space and time.",
    "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
    "The American Revolution had a profound impact on the birth of the United States as a nation.",
    "Regular exercise and a balanced diet are essential for maintaining good physical health.",
    "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
    "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
    "Startup companies often face challenges in securing funding and scaling their operations.",
    "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
]

genres = [
    "technology",
    "travel",
    "science",
    "food",
    "history",
    "fitness",
    "art",
    "climate change",
    "business",
    "music",
]

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],
    metadatas=[{"genre": g} for g in genres]
)

query_results = collection.query(
    query_texts=["Find me some delicious food!"],
    n_results=1,
)

query_results.keys()

query_results["documents"]

query_results["ids"]

query_results["distances"]

query_results["metadatas"]




