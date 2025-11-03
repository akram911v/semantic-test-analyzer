import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import logging

class Doc2VecModel:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.document_vectors = {}
        
    def train(self, tokenized_docs, documents):
        tagged_docs = [
            TaggedDocument(words=doc, tags=[str(i)]) 
            for i, doc in enumerate(tokenized_docs)
        ]
        
        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            dm=1  # PV-DM mode
        )
        
        for doc_id in range(len(tokenized_docs)):
            self.document_vectors[doc_id] = self.model.dv[str(doc_id)]
                
    def get_document_similarity(self, doc1_id, doc2_id):
        vec1 = self.document_vectors.get(doc1_id)
        vec2 = self.document_vectors.get(doc2_id)
        if vec1 is not None and vec2 is not None:
            return cosine_similarity([vec1], [vec2])[0][0]
        return 0.0
    
    def find_similar_documents(self, query_tokens, documents, top_n=5):
        if not self.model:
            raise ValueError("Model not trained")
        
        query_vector = self.model.infer_vector(query_tokens)
        similarities = []
        
        for doc_id, doc_vector in self.document_vectors.items():
            sim = cosine_similarity([query_vector], [doc_vector])[0][0]
            similarities.append((doc_id, sim, documents[doc_id]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_similar_documents_by_id(self, doc_id, documents, top_n=5):
        if not self.model:
            raise ValueError("Model not trained")
            
        similar_docs = self.model.dv.most_similar(str(doc_id), topn=top_n)
        results = []
        for similar_doc_id, similarity in similar_docs:
            doc_idx = int(similar_doc_id)
            results.append((doc_idx, similarity, documents[doc_idx]))
        return results
