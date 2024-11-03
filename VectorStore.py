import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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


