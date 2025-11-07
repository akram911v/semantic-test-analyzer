```markdown
# Extended Semantic Analyzer

This project extends the LSA-based semantic analyzer from Practical Work 2 with Word2Vec and Doc2Vec models for document similarity analysis, as required by Practical Work 3.

## Features

- **LSA (Latent Semantic Analysis)**: Original implementation from PW2 using TF-IDF and SVD
- **Word2Vec**: Extended implementation using simplified word embeddings
- **Doc2Vec**: Extended implementation using document embeddings
- **Comparative Analysis**: Side-by-side comparison of all three methods
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
   pip install numpy scikit-learn nltk pandas
   ```

3. Run the analyzer:
   ```bash
   python lsa_analyzer.py
   ```

## Usage

### Basic Usage
```python
from lsa_analyzer import LSASemanticAnalyzer

# Initialize analyzer
analyzer = LSASemanticAnalyzer(n_components=5)

# Train on documents
documents = ["doc1 text", "doc2 text", ...]
analyzer.fit(documents)

# Train extended models
analyzer.train_word2vec()
analyzer.train_doc2vec()

# Find similar documents
results = analyzer.query_similarity("your query", top_k=3)
```

### Comparative Analysis Example
The extended analyzer provides comparison between three semantic analysis methods:

```python
# LSA similarity
lsa_similarity = analyzer.document_similarity(doc1, doc2)

# Word2Vec similarity  
w2v_similarity = analyzer.word2vec_model.get_document_similarity(doc1_id, doc2_id)

# Doc2Vec similarity
d2v_similarity = analyzer.doc2vec_model.get_document_similarity(doc1_id, doc2_id)
```

## Method Comparison

### LSA (Latent Semantic Analysis)
- **Approach**: Matrix factorization (SVD) of TF-IDF matrix
- **Strengths**: Captures broad semantic topics, handles synonymy
- **Use Cases**: Document classification, topic modeling

### Word2Vec (Simplified)
- **Approach**: Word embeddings with document vectors as averages
- **Strengths**: Captures semantic word relationships
- **Use Cases**: Semantic similarity, word analogies

### Doc2Vec (Simplified) 
- **Approach**: Document embeddings using TF-IDF and dimensionality reduction
- **Strengths**: Direct document representation
- **Use Cases**: Document similarity, information retrieval

## Example Output
```
LSA Similarity: 0.7132
Word2Vec Similarity: 0.8554  
Doc2Vec Similarity: 0.9706
```

## Project Structure
```
semantic-test-analyzer/
├── lsa_analyzer.py      # Main analyzer with all three methods
├── word2vec_model.py    # Word2Vec implementation
├── doc2vec_model.py     # Doc2Vec implementation
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Requirements
- Python 3.7+
- numpy
- scikit-learn
- nltk
- pandas

## License
Academic Use - Practical Work 3 Submission
```

