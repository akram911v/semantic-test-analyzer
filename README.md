```markdown
# Extended Semantic Analyzer

This project extends the semantic analyzer with multiple approaches for document processing and similarity analysis, including LSA, Word2Vec, Doc2Vec, CNN, and RNN models.

## Features

- **LSA (Latent Semantic Analysis)**: Original implementation using TF-IDF and SVD
- **Word2Vec**: Word embeddings for semantic analysis
- **Doc2Vec**: Document embeddings for similarity comparison
- **CNN (Convolutional Neural Network)**: Neural network for pattern detection in text
- **RNN (Recurrent Neural Network)**: Neural network for sequential text processing
- **Comparative Analysis**: Side-by-side comparison of all five methods
- **Document Similarity**: Calculate semantic similarity between documents
- **Query Search**: Find most similar documents to a query

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

3. Run the analyzer:
   ```bash
   python main.py
   ```

## Usage

### Basic Usage
```python
from lsa_analyzer import LSASemanticAnalyzer
from word2vec_model import Word2VecModel
from doc2vec_model import Doc2VecModel
from cnn_analyzer import CNNSemanticAnalyzer
from rnn_analyzer import RNNSemanticAnalyzer

# Initialize all analyzers
lsa_analyzer = LSASemanticAnalyzer(n_components=5)
word2vec_model = Word2VecModel()
doc2vec_model = Doc2VecModel()
cnn_analyzer = CNNSemanticAnalyzer()
rnn_analyzer = RNNSemanticAnalyzer()

# Train on documents (for LSA, Word2Vec, Doc2Vec)
documents = ["doc1 text", "doc2 text", ...]
lsa_analyzer.fit(documents)
word2vec_model.train(documents)
doc2vec_model.train(documents)

# Note: CNN and RNN models require additional training data
```

### Comparative Analysis Example
The extended analyzer provides comparison between five semantic analysis methods:

```python
# LSA similarity
lsa_similarity = lsa_analyzer.document_similarity(doc1, doc2)

# Word2Vec similarity  
w2v_similarity = word2vec_model.get_document_similarity(doc1, doc2)

# Doc2Vec similarity
d2v_similarity = doc2vec_model.get_document_similarity(doc1, doc2)

# CNN embedding (requires trained model)
cnn_embedding = cnn_analyzer.get_document_embedding(doc1)

# RNN embedding (requires trained model)
rnn_embedding = rnn_analyzer.get_document_embedding(doc1)
```

## Method Comparison

### LSA (Latent Semantic Analysis)
- **Approach**: Matrix factorization (SVD) of TF-IDF matrix
- **Strengths**: Captures broad semantic topics, handles synonymy
- **Use Cases**: Document classification, topic modeling

### Word2Vec
- **Approach**: Word embeddings with document vectors as averages
- **Strengths**: Captures semantic word relationships
- **Use Cases**: Semantic similarity, word analogies

### Doc2Vec
- **Approach**: Document embeddings using TF-IDF and dimensionality reduction
- **Strengths**: Direct document representation
- **Use Cases**: Document similarity, information retrieval

### CNN (Convolutional Neural Network)
- **Approach**: 1D convolutional layers for pattern detection
- **Strengths**: Captures local word patterns and phrases
- **Use Cases**: Text classification, pattern recognition

### RNN (Recurrent Neural Network)
- **Approach**: Sequential processing with internal memory
- **Strengths**: Handles variable-length sequences, maintains context
- **Use Cases**: Sequential data analysis, context-dependent tasks

## Comparative Analysis Summary

| Method | Architecture | Strengths | Best For |
|--------|-------------|-----------|----------|
| LSA | Matrix Factorization | Fast, interpretable | Small-medium corpora |
| Word2Vec | Word Embeddings | Word-level semantics | Word similarity |
| Doc2Vec | Document Embeddings | Document-level representation | Document similarity |
| CNN | Convolutional Neural Net | Pattern detection | Classification |
| RNN | Recurrent Neural Net | Sequential context | Contextual analysis |

## Project Structure
```
semantic-test-analyzer/
├── main.py              # Main script with all five analyzers
├── lsa_analyzer.py      # LSA implementation
├── word2vec_model.py    # Word2Vec implementation
├── doc2vec_model.py     # Doc2Vec implementation
├── cnn_analyzer.py      # CNN neural network implementation
├── rnn_analyzer.py      # RNN neural network implementation
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Requirements
- Python 3.7+
- numpy
- scikit-learn
- nltk
- pandas
- tensorflow

## License
Academic Use - Practical Work 4 Submission
```
