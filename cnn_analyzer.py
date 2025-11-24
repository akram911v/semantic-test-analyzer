import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class CNNSemanticAnalyzer:
    def __init__(self):
        self.model = self._build_cnn_model()
    
    def _build_cnn_model(self):
        model = tf.keras.Sequential([
            layers.Embedding(input_dim=10000, output_dim=128),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')  # Document embedding
        ])
        return model
    
   def get_document_embedding(self, processed_text):
    # Convert text to sequence (simple example)
    # In real implementation, you'd use your tokenizer from Assignment #3
    sequence = [1, 2, 3, 4, 5]  # Replace with actual token indices
    sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=100)
    embedding = self.model.predict(sequence, verbose=0)
    return embedding[0]  # Return the document embedding vector
