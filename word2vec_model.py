
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import logging

class Word2VecModel:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.document_vectors = {}
        
    def train(self, tokenized_docs, documents):
        self.model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1
        )
        
        for doc_id, tokens in enumerate(tokenized_docs):
            word_vectors = []
            for word in tokens:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
            if word_vectors:
                self.document_vectors[doc_id] = np.mean(word_vectors, axis=0)
            else:
                self.document_vectors[doc_id] = np.zeros(self.vector_size)
                
    def get_document_similarity(self, doc1_id, doc2_id):
        vec1 = self.document_vectors.get(doc1_id)
        vec2 = self.document_vectors.get(doc2_id)
        if vec1 is not None and vec2 is not None:
            return cosine_similarity([vec1], [vec2])[0][0]
        return 0.0
