
```markdown
# Extended Semantic Analyzer with Chatbot System

This project implements a comprehensive semantic analysis and text generation system across multiple practical works (PW2-PW6). The system includes semantic analyzers, text generators, and an autonomous dialog system.

## Features

### Semantic Analyzers (PW2-PW4):
- **LSA (Latent Semantic Analysis)**: Matrix factorization for semantic topic extraction
- **Word2Vec**: Word embeddings for semantic word relationships
- **Doc2Vec**: Document embeddings for similarity comparison
- **CNN (Convolutional Neural Network)**: Pattern detection in text sequences
- **RNN (Recurrent Neural Network)**: Sequential text processing with memory

### Text Generators (PW5-PW6):
#### Template-Based Generators (PW5):
- **LSTM Template Generator**: Long Short-Term Memory with predefined templates
- **Seq2Seq Template Generator**: Encoder-decoder architecture with template filling

#### Non-Template Generators (PW6):
- **Markov Chain Generator**: Probability-based text generation using n-grams
- **Character-Level RNN Generator**: Character-by-character text generation
- **Advanced Markov Generator**: Enhanced Markov chains for better structure
- **Chatbot Generator**: Dialog-focused response generation

### Dialog System (PW6):
- **Chatbot System**: Autonomous dialog system with multiple generation methods
- **Interactive Chat Interface**: Real-time conversation capabilities
- **Method Comparison**: Template vs non-template generation analysis

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

## Usage

### Quick Start
```bash
# Run Assignment 5 demonstration (Template-based generation)
python main.py

# Run Assignment 6 demonstration (Non-template generation and chatbot)
python chatbot_demo.py
```

### Semantic Analysis
```python
from lsa_analyzer import LSASemanticAnalyzer
from cnn_analyzer import CNNSemanticAnalyzer

lsa = LSASemanticAnalyzer()
cnn = CNNSemanticAnalyzer()

# Train on documents
documents = ["text1", "text2", "text3"]
lsa.fit(documents)

# Calculate similarity
similarity = lsa.document_similarity("text1", "text2")
```

### Template-Based Generation (PW5)
```python
from lstm_generator import LSTMTemplateGenerator
from seq2seq_generator import Seq2SeqGenerator

# Initialize generators
lstm_gen = LSTMTemplateGenerator()
seq2seq_gen = Seq2SeqGenerator()

# Generate from templates
lstm_text = lstm_gen.generate_from_template()
seq2seq_text = seq2seq_gen.generate_from_template()
```

### Non-Template Generation (PW6)
```python
from markov_generator import MarkovChainGenerator
from char_rnn_generator import CharRNNGenerator

# Initialize and train Markov generator
markov_gen = MarkovChainGenerator()
markov_gen.train(["training text 1", "training text 2"])

# Generate sentences
sentence = markov_gen.generate_sentence()

# Character-level generation
char_rnn = CharRNNGenerator()
char_rnn.prepare_text("training text")
generated = char_rnn.generate_text("seed", length=100)
```

### Chatbot System (PW6)
```python
from chatbot_system import ChatbotSystem
from lstm_generator import LSTMTemplateGenerator
from markov_generator import MarkovChainGenerator

# Create chatbot system
chatbot = ChatbotSystem(name="MyChatBot")

# Add generators
lstm_gen = LSTMTemplateGenerator()
markov_gen = MarkovChainGenerator()
markov_gen.train(["training data"])

chatbot.add_generator("LSTM Template", lstm_gen, "template")
chatbot.add_generator("Markov Chain", markov_gen, "non_template")

# Get responses
response = chatbot.get_response("Hello, how are you?")
print(response)

# Start interactive chat
chatbot.start_chat_session()
```

## Examples

### Semantic Similarity Example
```
Document 1: "Machine learning is a subset of artificial intelligence"
Document 2: "AI and machine learning are transforming technology"

LSA Similarity: 0.7762
CNN Similarity: 0.7529
RNN Similarity: 0.7883
```

### Template-Based Generation Example
```
Template: "The [ADJECTIVE] [NOUN] [VERB] [ADVERB] in the context of [TOPIC]"
Generated: "The important analysis demonstrates clearly in the context of machine learning"
```

### Non-Template Generation Example
```
Input: "What is machine learning?"
Generated: "Machine learning enables systems to learn from data patterns and make predictions based on neural networks and algorithms."
```

### Chatbot Interaction Example
```
You: What is natural language processing?
Bot: Natural language processing helps computers understand human language through semantic analysis and pattern recognition.
```

## Comparative Analysis

### Template vs Non-Template Generation

| Aspect | Template-Based (PW5) | Non-Template (PW6) |
|--------|---------------------|-------------------|
| **Approach** | Predefined templates with slots | Pattern learning from data |
| **Creativity** | Limited to template structure | High, generates novel combinations |
| **Grammar** | Guaranteed correct | Learned from training data |
| **Variety** | Limited by templates | High variation possible |
| **Control** | High control over output | Less control, more organic |
| **Use Case** | Structured responses, forms | Natural conversation, creative writing |

### Generator Comparison

| Generator | Type | Strengths | Best For |
|-----------|------|-----------|----------|
| **LSTM Template** | Template | Structured, grammatical | Form-based responses |
| **Seq2Seq Template** | Template | Context-aware templates | Question-answer pairs |
| **Markov Chain** | Non-Template | Simple, probabilistic | Short text generation |
| **Character RNN** | Non-Template | Character-level patterns | Creative text, names |
| **Chatbot Generator** | Non-Template | Dialog-focused | Conversation responses |

## Project Structure
```
semantic-test-analyzer/
├── main.py              # PW5: Template-based generation demo
├── chatbot_demo.py      # PW6: Chatbot system demo
├── chatbot_system.py    # PW6: Main chatbot system
├── lsa_analyzer.py      # LSA semantic analyzer
├── word2vec_model.py    # Word2Vec implementation
├── doc2vec_model.py     # Doc2Vec implementation
├── cnn_analyzer.py      # CNN semantic analyzer
├── rnn_analyzer.py      # RNN semantic analyzer
├── lstm_generator.py    # PW5: LSTM template generator
├── seq2seq_generator.py # PW5: Seq2Seq template generator
├── markov_generator.py  # PW6: Markov chain generator
├── char_rnn_generator.py # PW6: Character-level RNN generator
├── requirements.txt     # Dependencies
└── README.md           # This documentation
```

## Practical Work Progression

### PW2: Basic semantic analyzer (LSA)
- Document similarity using Latent Semantic Analysis
- TF-IDF and SVD implementation

### PW3: Extended with Word2Vec and Doc2Vec
- Word embeddings for semantic relationships
- Document embeddings for similarity

### PW4: Extended with CNN and RNN neural networks
- Convolutional neural networks for text patterns
- Recurrent neural networks for sequence processing

### PW5: Template-based text generation
- LSTM with template filling
- Sequence-to-sequence template generation
- Template-based approach only

### PW6: Non-template text generation and chatbot
- Markov chain text generation
- Character-level RNN generation
- Autonomous dialog system
- Comparison of template vs non-template approaches

## Requirements
- Python 3.7+
- numpy
- scikit-learn
- nltk
- pandas
- tensorflow (for some neural network implementations)

## Key Achievements

### Assignment 5 (Template-Based Generation):
- ✅ Implemented LSTM neural networks with templates
- ✅ Implemented sequence-to-sequence methods with templates
- ✅ Template-based generation only (as required)
- ✅ Integration with semantic analyzers

### Assignment 6 (Non-Template Generation and Chatbot):
- ✅ Developed autonomous dialog system (chatbot)
- ✅ Implemented non-template text generation methods
- ✅ Markov chain and character-level RNN generators
- ✅ Comparison between template and non-template approaches
- ✅ Interactive chat interface
- ✅ Analysis of generation method differences

## License
Academic Use - Practical Works 2-6 Submission
```
```

