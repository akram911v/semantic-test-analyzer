import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecModel:
    def __init__(self, vector_size=100, window=2, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.word_vectors = {}
        self.document_vectors = {}
        self.vocab = []
        
    def train(self, tokenized_docs, documents):
        # Build vocabulary
        word_freq = {}
        for doc in tokenized_docs:
            for word in doc:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Create vocabulary list
        self.vocab = []
        for word, freq in word_freq.items():
            if freq >= self.min_count:
                self.vocab.append(word)
        
        # Initialize random word vectors
        np.random.seed(42)
        for word in self.vocab:
            self.word_vectors[word] = np.random.normal(0, 0.1, self.vector_size)
        
        # Simple co-occurrence based training
        for doc in tokenized_docs:
            for i, target_word in enumerate(doc):
                if target_word not in self.word_vectors:
                    continue
                    
                # Context window
                start = max(0, i - self.window)
                end = min(len(doc), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j and doc[j] in self.word_vectors:
                        # Simple update rule
                        self.word_vectors[target_word] += 0.01 * self.word_vectors[doc[j]]
        
        # Create document vectors
        for doc_id, tokens in enumerate(tokenized_docs):
            word_vecs = []
            for word in tokens:
                if word in self.word_vectors:
                    word_vecs.append(self.word_vectors[word])
            
            if word_vecs:
                self.document_vectors[doc_id] = np.mean(word_vecs, axis=0)
            else:
                self.document_vectors[doc_id] = np.zeros(self.vector_size)
                
    def get_document_similarity(self, doc1_id, doc2_id):
        vec1 = self.document_vectors.get(doc1_id)
        vec2 = self.document_vectors.get(doc2_id)
        if vec1 is not None and vec2 is not None:
            return cosine_similarity([vec1], [vec2])[0][0]
        return 0.0

    def find_similar_documents(self, query_tokens, documents, top_n=5):
        query_vector = self._get_document_vector(query_tokens)
        similarities = []
        
        for doc_id, doc_vector in self.document_vectors.items():
            sim = cosine_similarity([query_vector], [doc_vector])[0][0]
            similarities.append((doc_id, sim, documents[doc_id]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def _get_document_vector(self, tokens):
        word_vectors = []
        for word in tokens:
            if word in self.word_vectors:
                word_vectors.append(self.word_vectors[word])
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        return np.zeros(self.vector_size)
