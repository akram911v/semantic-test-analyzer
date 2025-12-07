import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

class LSTMTemplateGenerator:
    def __init__(self, max_sequence_len=50, embedding_dim=100, lstm_units=128):
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.tokenizer = None
        self.templates = [
            "The [ADJECTIVE] [NOUN] [VERB] [ADVERB] in the context of [TOPIC]",
            "[ENTITY] is known for [ACTION] in the field of [FIELD]",
            "When considering [CONCEPT], we observe that [OBSERVATION]",
            "The relationship between [SUBJECT] and [OBJECT] demonstrates [PATTERN]",
            "In [DOMAIN], the [COMPONENT] plays a crucial role in [FUNCTION]"
        ]
        self.vocab_categories = {
            'ADJECTIVE': ['important', 'significant', 'relevant', 'critical', 'essential'],
            'NOUN': ['analysis', 'system', 'method', 'approach', 'model'],
            'VERB': ['demonstrates', 'shows', 'indicates', 'suggests', 'reveals'],
            'ADVERB': ['clearly', 'effectively', 'efficiently', 'precisely', 'accurately'],
            'TOPIC': ['machine learning', 'natural language processing', 'data analysis', 'artificial intelligence'],
            'ENTITY': ['Deep learning', 'Neural networks', 'AI systems', 'Computational models'],
            'ACTION': ['pattern recognition', 'feature extraction', 'data processing', 'knowledge representation'],
            'FIELD': ['computer science', 'data science', 'cognitive science', 'information theory'],
            'CONCEPT': ['semantic analysis', 'text generation', 'language modeling', 'context understanding'],
            'OBSERVATION': ['meaningful patterns emerge', 'contextual relationships form', 'semantic connections appear'],
            'SUBJECT': ['input data', 'feature vectors', 'training examples', 'semantic representations'],
            'OBJECT': ['output predictions', 'classification results', 'generated text', 'analytical insights'],
            'PATTERN': ['predictive relationships', 'correlational structures', 'causal connections', 'hierarchical organization'],
            'DOMAIN': ['semantic analysis', 'text generation', 'language processing', 'information retrieval'],
            'COMPONENT': ['LSTM layer', 'attention mechanism', 'embedding matrix', 'sequence generator'],
            'FUNCTION': ['context preservation', 'sequence generation', 'pattern learning', 'semantic understanding']
        }
    
    def build_model(self, vocab_size):
        """Build LSTM model for text generation"""
        self.model = tf.keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=self.max_sequence_len-1),
            layers.LSTM(self.lstm_units, return_sequences=True),
            layers.LSTM(self.lstm_units),
            layers.Dense(self.lstm_units, activation='relu'),
            layers.Dense(vocab_size, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return self.model
    
    def generate_from_template(self, template=None):
        """Generate text based on template"""
        if template is None:
            template = random.choice(self.templates)
        
        # Fill template slots
        generated_text = template
        for slot in self.vocab_categories.keys():
            if f"[{slot}]" in generated_text:
                word = random.choice(self.vocab_categories[slot])
                generated_text = generated_text.replace(f"[{slot}]", word)
        
        return generated_text
    
    def generate_sequence(self, seed_text, length=20):
        """Generate a sequence of text from seed (simulated for now)"""
        # In a full implementation, this would use the trained LSTM
        # For now, we'll return a simulated generated text
        base_sentences = [
            "The semantic analysis reveals important patterns in the data.",
            "Natural language processing enables computers to understand human language.",
            "Machine learning models can generate coherent text based on training data.",
            "The LSTM network effectively captures long-term dependencies in sequences.",
            "Sequence generation requires understanding of context and semantics."
        ]
        
        return f"{seed_text} " + random.choice(base_sentences)
    
    def get_template_slots(self, template):
        """Extract slots from a template"""
        import re
        slots = re.findall(r'\[(.*?)\]', template)
        return slots
    
    def get_available_templates(self):
        """Return list of available templates"""
        return self.templates
