# 🎬 IMDB Movie Sentiment Analysis

An end-to-end Deep Learning and Natural Language Processing (NLP) project that classifies movie reviews as **Positive** or **Negative**. The project implements a recurrent neural network trained on the IMDB dataset, paired with an interactive **Streamlit** web application for real-time sentiment inference.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Model Architecture & Training](#-model-architecture--training)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Notebook Workflows](#-notebook-workflows)
- [🔮 Future Aspects & Accuracy Improvements](#-future-aspects--accuracy-improvements)
  - [1. Implementing LSTM and GRU Architectures](#1-implementing-lstm-and-gru-architectures)
  - [2. Leveraging Transformer-Based & Contextual Embedding Models](#2-leveraging-transformer-based--contextual-embedding-models)
  - [3. Additional Engineering & Data Enhancements](#3-additional-engineering--data-enhancements)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 📖 Overview

Sentiment analysis of text is a foundational NLP task with wide practical applications in audience feedback, product reviews, and recommendation systems. This project demonstrates:
1. Preprocessing raw text reviews into numerical token sequences.
2. Training a deep neural network featuring an **Embedding Layer** and a **SimpleRNN (Recurrent Neural Network)**.
3. Preventing overfitting using early stopping callbacks.
4. Exporting the trained model (`.h5`) and serving it via a clean **Streamlit** web user interface.

---

## ✨ Key Features

- **End-to-End Pipeline**: From raw text preprocessing, vocabulary mapping, and sequence padding to deep learning inference.
- **Deep Recurrent Network**: Sequential modeling with an Embedding layer and SimpleRNN with ReLU activation.
- **Interactive Web App**: User-friendly Streamlit dashboard allowing users to input any movie review and receive an instant sentiment prediction and confidence score.
- **Hands-on Jupyter Notebooks**: Detailed notebooks for data exploration, model training, and prediction prototyping.

---

## 📊 Dataset

The model is trained on the widely benchmarked **IMDB Movie Reviews Dataset**:
- **Total Reviews**: 50,000 highly polarized movie reviews (25,000 for training, 25,000 for testing).
- **Label Distribution**: Balanced (50% positive reviews, 50% negative reviews).
- **Vocabulary Size**: Configured to top `10,000` most frequent words.
- **Sequence Length**: Fixed to `maxlen=500` tokens via sequence padding (`pad_sequences`).
- **Token Index Offset**: Accommodates standard reserved indices (`0: padding`, `1: start of sequence`, `2: unknown/out-of-vocabulary`, `3: unused`).

---

## 🧠 Model Architecture & Training

The baseline neural network is built with **Keras / TensorFlow**:

```text
Input (Review Tokens, shape: [None, 500])
         │
         ▼
┌───────────────────────────────────────────┐
│ Embedding Layer                           │
│ - Vocabulary: 10,000 words                │
│ - Embedding Dimension: 128                │
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│ SimpleRNN Layer                           │
│ - Hidden Units: 128                       │
│ - Activation: ReLU                        │
└───────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│ Dense Output Layer                        │
│ - Units: 1                                │
│ - Activation: Sigmoid                     │
└───────────────────────────────────────────┘
         │
         ▼
Output: Sentiment Probability [0.0 - 1.0]
(Score > 0.5: Positive, Score <= 0.5: Negative)
```

### Training Configuration
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `adam`
- **Batch Size**: 32
- **Validation Split**: 20%
- **Callback**: `EarlyStopping` monitoring `val_loss` with patience of 5 epochs and best weights restoration.
- **Baseline Performance**: Achieves ~84–85% validation accuracy.

---

## 📂 Project Structure

```bash
movie-sentiment-analysis/
├── app.py                             # Streamlit web app for real-time inference
├── movie_sentiment_analysis_model.h5  # Trained Keras model weights and architecture
├── project.ipynb                      # Main notebook: EDA, preprocessing, model training & export
├── prediction.ipynb                   # Notebook: Model loading and prediction testing
├── experiments.ipynb                  # Experimental notebook: One-hot encoding & NLP experiments
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git ignore rules (venv, IDE configs, etc.)
└── README.md                          # Comprehensive project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ (Python 3.10 or 3.11 recommended)
- `pip` package manager
- Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/krishnanamdev558/deep-learning-projects.git
cd deep-learning-projects/projects/movie-sentiment-analysis
```

### Step 2: Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Launch the Streamlit interactive web application with:

```bash
streamlit run app.py
```

Once running, open your browser at `http://localhost:8501`. Enter any custom movie review (e.g., *"The acting was phenomenal and the cinematography was stunning!"*), click **Classify**, and view:
- **Sentiment**: `Positive` or `Negative`
- **Prediction Score**: Confidence probability between `0.0` and `1.0`.

---

## 📓 Notebook Workflows

- **`project.ipynb`**: Demonstrates the end-to-end training pipeline. Loads the IMDB dataset, builds the Sequential model, compiles, trains with early stopping, evaluates performance, and exports `movie_sentiment_analysis_model.h5`.
- **`prediction.ipynb`**: Demonstrates how to restore the saved `.h5` model, preprocess arbitrary text strings using `imdb.get_word_index()`, pad sequences to 500 tokens, and run model inference.
- **`experiments.ipynb`**: Scratchpad exploring text tokenization, one-hot encoding representations, and vocabulary indexing.

---

## 🔮 Future Aspects & Accuracy Improvements

While the baseline SimpleRNN model provides a solid starting point (~85% validation accuracy), standard SimpleRNNs struggle with long text documents due to vanishing gradients and limited memory retention. The following architectural upgrades and techniques are planned to significantly boost classification accuracy and generalization.

### 1. Implementing LSTM and GRU Architectures

Standard SimpleRNN layers suffer from vanishing and exploding gradients when processing long sequences (such as 500-word reviews). To overcome this:

- **Long Short-Term Memory (LSTM)**:
  - Incorporates dedicated memory cells and gating mechanisms (**forget gate, input gate, output gate**).
  - Preserves long-range contextual dependencies across lengthy reviews, ensuring critical sentiment clues early or late in the review are not lost.
  - Expected accuracy improvement: **~88%–90%**.
- **Bidirectional LSTM (BiLSTM)**:
  - Traverses review sequences in both forward (past $\to$ future) and backward (future $\to$ past) directions.
  - Captures full contextual meaning of words where sentiment depends on surrounding context (e.g., *"not"* followed several words later by *"entertaining"*).
- **Gated Recurrent Unit (GRU)**:
  - Merges the cell state and hidden state using two gates (**reset gate and update gate**).
  - Offers comparable expressive power to LSTMs with fewer trainable parameters, enabling faster convergence and lower computational overhead.
- **Regularization (Dropout & SpatialDropout1D)**:
  - Introduce `SpatialDropout1D(0.2)` after the embedding layer and recurrent dropout inside LSTM/GRU layers to close the gap between training accuracy (~95%) and validation accuracy (~85%).

### 2. Leveraging Transformer-Based & Contextual Embedding Models

The current model learns static token embeddings from scratch. Transitioning to modern transformer-based language representations will unlock state-of-the-art sentiment understanding:

- **Contextual Pre-Trained Transformers (BERT / DistilBERT / RoBERTa)**:
  - Models like **BERT (Bidirectional Encoder Representations from Transformers)** or lightweight variants like **DistilBERT** understand deep semantic nuances, sarcasm, double negations, and colloquial language.
  - Pre-trained on massive corpora, fine-tuning them on IMDB easily yields **93%–96%+ accuracy**.
- **Pre-trained Embeddings (Word2Vec / GloVe / FastText)**:
  - As an intermediate upgrade, initialize the embedding layer with pre-trained 300-dimensional GloVe vectors instead of training embeddings from random initialization.
  - Captures universal semantic similarities across words even when training data is limited.
- **Sentence Transformers & Modern NLP Pipelines**:
  - Integrate Hugging Face `transformers` pipeline (`AutoTokenizer`, `AutoModelForSequenceClassification`).
  - Export models with ONNX Runtime or TensorRT for optimized low-latency inference in production.

### 3. Additional Engineering & Data Enhancements

- **Text Cleaning & Preprocessing Enhancements**:
  - Strip HTML tags (`<br />`), normalize contractions (e.g., *"don't"* $\to$ *"do not"*), and handle punctuation more granularly prior to tokenization.
- **Hyperparameter Optimization**:
  - Systematic tuning of learning rates (with cosine decay / warm-up schedules), embedding dimensions (128 vs 256), and sequence lengths.
- **Modern Model Formats**:
  - Migrate from the legacy Keras `.h5` format to the modern Keras native `.keras` format for improved serialization and compatibility.
- **Containerization & Cloud Deployment**:
  - Create a `Dockerfile` for containerized deployment to platforms such as Hugging Face Spaces, Render, or AWS ECS.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Data Processing & Math** | NumPy |
| **Web Framework** | Streamlit |
| **Interactive Environment** | Jupyter Notebook / IPyKernel |
| **Dataset** | Keras Built-in IMDB Dataset |

---

## 👤 Author

- **Krishna Namdev**
- GitHub: [@krishnanamdev558](https://github.com/krishnanamdev558)
