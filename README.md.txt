# Semantic Text Analyzer using LSA 

  

This project implements a semantic text analyzer using Latent Semantic Analysis (LSA), building upon the text preprocessing techniques from Practical Work 1. 

  

## Features 

  

- **Text Preprocessing**: Tokenization, stopword removal, lemmatization 

- **TF-IDF Vectorization**: Convert text to numerical representations 

- **Latent Semantic Analysis**: Dimensionality reduction using SVD 

- **Semantic Similarity**: Calculate similarity between documents 

- **Query Search**: Find most semantically similar documents 

- **Topic Analysis**: Extract meaningful topics from document collection 

  

## Installation 

  

1. Clone the repository: 

```bash 

git clone https://github.com/akram911v/semantic-test-analyzer.git 

cd semantic-test-analyzer 

``` 

  

2. Install dependencies: 

```bash 

pip install -r requirements.txt 

``` 

  

3. Run the analyzer: 

```bash 

python lsa_analyzer.py 

``` 

  

## Usage Examples 

  

### Basic Usage 

```python 

from lsa_analyzer import LSASemanticAnalyzer 

  

# Initialize analyzer 

analyzer = LSASemanticAnalyzer(n_components=50) 

  

# Train on documents 

documents = ["your text documents here..."] 

analyzer.fit(documents) 

  

# Find similar documents 

results = analyzer.query_similarity("machine learning", top_k=3) 

``` 

  

### Document Similarity 

```python 

similarity = analyzer.document_similarity( 

    "Machine learning algorithms", 

    "Artificial intelligence systems" 

) 

``` 

  

## Project Structure 

  

- `lsa_analyzer.py` - Main LSA implementation 

- `requirements.txt` - Project dependencies 

- `README.md` - This file 

  

## Technical Details 

  

- Uses TF-IDF for document representation 

- Implements Truncated SVD for dimensionality reduction 

- Cosine similarity for semantic comparison 

- NLTK for text preprocessing 

  

## Results 

  

The analyzer can: 

- Discover latent topics in document collections 

- Measure semantic similarity between texts 

- Perform semantic search queries 

- Handle various text domains and sizes  