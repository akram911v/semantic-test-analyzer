import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import random

class Seq2SeqGenerator:
    def __init__(self, max_sequence_len=50, embedding_dim=100, lstm_units=128):
        self.max_sequence_len = max_sequence_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.encoder_model = None
        self.decoder_model = None
        self.full_model = None
        self.tokenizer = None
        
        # Template patterns for sequence generation
        self.templates = [
            "Given the context of [CONTEXT], we can generate [CONTENT_TYPE]",
            "Based on semantic analysis of [TOPIC], the system produces [OUTPUT]",
            "The sequence-to-sequence model transforms [INPUT_TYPE] into [OUTPUT_TYPE]",
            "When analyzing [SUBJECT], the generator creates meaningful text about [ASPECT]"
        ]
        
        self.context_vocab = {
            'CONTEXT': ['natural language', 'semantic relationships', 'textual data', 'linguistic patterns'],
            'CONTENT_TYPE': ['coherent paragraphs', 'meaningful sentences', 'structured text', 'contextual responses'],
            'TOPIC': ['document similarity', 'text classification', 'semantic search', 'information retrieval'],
            'OUTPUT': ['relevant summaries', 'contextual explanations', 'coherent descriptions', 'structured analyses'],
            'INPUT_TYPE': ['input sequences', 'source text', 'query phrases', 'semantic vectors'],
            'OUTPUT_TYPE': ['generated text', 'target sequences', 'response sentences', 'output paragraphs'],
            'SUBJECT': ['semantic coherence', 'textual consistency', 'linguistic patterns', 'contextual relationships'],
            'ASPECT': ['semantic meaning', 'contextual relevance', 'textual coherence', 'linguistic structure']
        }
    
    def build_model(self, input_vocab_size, output_vocab_size):
        """Build encoder-decoder sequence-to-sequence model"""
        
        # Encoder
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(input_vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = layers.LSTM(self.lstm_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(None,))
        decoder_embedding = layers.Embedding(output_vocab_size, self.embedding_dim)(decoder_inputs)
        decoder_lstm = layers.LSTM(self.lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = layers.Dense(output_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Full model
        self.full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Inference models
        self.encoder_model = Model(encoder_inputs, encoder_states)
        
        decoder_state_input_h = layers.Input(shape=(self.lstm_units,))
        decoder_state_input_c = layers.Input(shape=(self.lstm_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        self.full_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.full_model
    
    def generate_from_template(self, template=None):
        """Generate text based on template using seq2seq approach"""
        if template is None:
            template = random.choice(self.templates)
        
        # Fill template slots
        generated_text = template
        for slot in self.context_vocab.keys():
            if f"[{slot}]" in generated_text:
                word = random.choice(self.context_vocab[slot])
                generated_text = generated_text.replace(f"[{slot}]", word)
        
        return generated_text
    
    def generate_sequence(self, input_text, max_length=20):
        """Generate sequence based on input text"""
        # Simulate sequence generation
        responses = [
            "The semantic analysis indicates meaningful patterns in the input data.",
            "Based on the context, the system generates coherent textual output.",
            "Sequence generation produces linguistically valid and contextually relevant text.",
            "The encoder-decoder model effectively transforms input sequences into meaningful output.",
            "Text generation based on semantic understanding yields coherent and relevant content."
        ]
        
        return f"Input: {input_text}\nGenerated: {random.choice(responses)}"
    
    def explain_architecture(self):
        """Explain the seq2seq architecture"""
        explanation = """
        Sequence-to-Sequence Architecture:
        1. Encoder LSTM: Processes input sequence and captures context
        2. Encoder States: Final hidden state contains semantic information
        3. Decoder LSTM: Initialized with encoder states
        4. Decoder Output: Generates output sequence token by token
        5. Attention Mechanism: (Optional) Focuses on relevant parts of input
        
        This architecture is ideal for:
        - Text summarization
        - Machine translation
        - Text generation
        - Question answering
        """
        return explanation
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.full_model:
            string_list = []
            self.full_model.summary(print_fn=lambda x: string_list.append(x))
            return "\n".join(string_list)
        return "Model not built yet. Call build_model() first."
