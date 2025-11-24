import numpy as np

class RNNSemanticAnalyzer:
    def __init__(self):
        print("RNN Semantic Analyzer initialized")
        self.embedding_size = 64
    
    def get_document_embedding(self, processed_text):
        # Simulate RNN document embedding
        print(f"RNN: Generating embedding for document (simulated)")
        return np.random.rand(self.embedding_size)
    
    def document_similarity(self, doc1, doc2):
        embedding1 = self.get_document_embedding(doc1)
        embedding2 = self.get_document_embedding(doc2)
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity