```markdown
# Extended Semantic Analyzer with Text Generation

This project extends the semantic analyzer with multiple approaches for document processing, similarity analysis, and text generation. The system includes implementations from Practical Works 2-5.

## Features

### Semantic Analyzers (Practical Works 2-4):
- **LSA (Latent Semantic Analysis)**: Matrix factorization for semantic topic extraction
- **Word2Vec**: Word embeddings for semantic word relationships
- **Doc2Vec**: Document embeddings for similarity comparison
- **CNN (Convolutional Neural Network)**: Pattern detection in text sequences
- **RNN (Recurrent Neural Network)**: Sequential text processing with memory

### Text Generators (Practical Work 5):
- **LSTM Template Generator**: Long Short-Term Memory network for template-based text generation
- **Sequence-to-Sequence Generator**: Encoder-decoder architecture for coherent text generation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/akram911v/semantic-test-analyzer.git
   cd semantic-test-analyzer
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main system:
   ```bash
   python main.py
   ```

## Usage

### Semantic Analysis (Existing Functionality)
```python
from lsa_analyzer import LSASemanticAnalyzer
from cnn_analyzer import CNNSemanticAnalyzer
from rnn_analyzer import RNNSemanticAnalyzer

# Initialize analyzers
lsa = LSASemanticAnalyzer()
cnn = CNNSemanticAnalyzer()
rnn = RNNSemanticAnalyzer()

# Train on documents
documents = ["doc1 text", "doc2 text", ...]
lsa.fit(documents)

# Calculate similarity
similarity = lsa.document_similarity("text1", "text2")
cnn_similarity = cnn.document_similarity("text1", "text2")
rnn_similarity = rnn.document_similarity("text1", "text2")
```

### Text Generation (New for PW5)
```python
from lstm_generator import LSTMTemplateGenerator
from seq2seq_generator import Seq2SeqGenerator

# Initialize generators
lstm_gen = LSTMTemplateGenerator()
seq2seq_gen = Seq2SeqGenerator()

# Generate from templates
lstm_text = lstm_gen.generate_from_template()
seq2seq_text = seq2seq_gen.generate_from_template()

# Generate sequence from input
input_text = "semantic analysis"
generated = seq2seq_gen.generate_sequence(input_text)
```

## Examples

### Example 1: Semantic Similarity Analysis
```
Document 1: "Machine learning is a subset of artificial intelligence"
Document 2: "AI and machine learning are transforming technology"

LSA Similarity: 0.7132
CNN Similarity: 0.7954  
RNN Similarity: 0.7516
```

### Example 2: Template-Based Text Generation
```
Template: "The [ADJECTIVE] [NOUN] [VERB] [ADVERB] in the context of [TOPIC]"
Generated: "The important analysis demonstrates effectively in the context of machine learning"
```

### Example 3: Sequence-to-Sequence Generation
```
Input: "semantic analysis of documents"
Generated: "Input: semantic analysis of documents
Generated: The semantic analysis indicates meaningful patterns in the input data."
```

## Comparative Analysis

### Semantic Analyzers Comparison

| Method | Architecture | Strengths | Best Use Cases |
|--------|-------------|-----------|----------------|
| **LSA** | Matrix Factorization | Fast, interpretable, handles synonymy | Document classification, topic modeling |
| **Word2Vec** | Word Embeddings | Captures semantic word relationships | Word similarity, analogies |
| **Doc2Vec** | Document Embeddings | Direct document representation | Document similarity, retrieval |
| **CNN** | Convolutional Neural Net | Pattern detection, position-invariant | Text classification, phrase detection |
| **RNN** | Recurrent Neural Net | Sequential context, memory | Context-dependent analysis, sequences |

### Text Generators Comparison

| Method | Architecture | Approach | Strengths |
|--------|-------------|----------|-----------|
| **LSTM Generator** | Long Short-Term Memory | Template-based generation | Captures long-term dependencies, maintains context |
| **Seq2Seq Generator** | Encoder-Decoder LSTM | Sequence transformation | Handles variable-length sequences, end-to-end learning |

### System Integration

The system demonstrates a complete pipeline:
1. **Semantic Analysis**: Extract meaning and relationships from text
2. **Similarity Calculation**: Compare documents using multiple methods
3. **Text Generation**: Create meaningful text based on semantic understanding
4. **Template Filling**: Generate coherent text using predefined patterns

## Project Structure
```
semantic-test-analyzer/
├── main.py              # Main integration script
├── lsa_analyzer.py      # LSA implementation
├── word2vec_model.py    # Word2Vec implementation
├── doc2vec_model.py     # Doc2Vec implementation
├── cnn_analyzer.py      # CNN semantic analyzer
├── rnn_analyzer.py      # RNN semantic analyzer
├── lstm_generator.py    # LSTM text generator (PW5)
├── seq2seq_generator.py # Seq2Seq text generator (PW5)
├── requirements.txt     # Dependencies
└── README.md           # This documentation
```

## Requirements
- Python 3.7+
- numpy
- scikit-learn
- nltk
- pandas
- tensorflow (for CNN, RNN, LSTM, Seq2Seq models)

## Practical Work Progression

### PW2: Basic semantic analyzer (LSA)
### PW3: Extended with Word2Vec and Doc2Vec
### PW4: Extended with CNN and RNN neural networks
### PW5: Extended with LSTM and sequence-to-sequence text generation

## Key Achievements for Practical Work 5
1. ✅ Implemented LSTM neural networks (now permitted)
2. ✅ Implemented sequence-to-sequence methods (now permitted)
3. ✅ Template-based text generation only (as required)
4. ✅ Integration with existing 4 semantic analyzers
5. ✅ Meaningful text generation based on semantic analysis

## License
Academic Use - Practical Work 5 Submission
```
