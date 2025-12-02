

## LSTM VS TRANSFORMER: Predicting Stock Movement Using News Embeddings
# Research Question

Do text embeddings from an LSTM with Attention or a Transformer built from scratch provide better predictive power for next-day stock movement when trained on financial news headlines?

We specifically investigate whether the type of text embedding (LSTM-based vs Transformer-based) affects the accuracy of Up/Down stock prediction.

# Motivation

Financial markets react instantly to news. Analysts, traders, and machine learning engineers all attempt to model how headlines influence next-day stock prices.
While many studies use pretrained models like FinBERT, our project asks a more fundamental question:

What happens when we remove pretraining and train both architectures from scratch on the same dataset?

By doing so, we isolate the impact of architecture alone (LSTM vs Transformer) while holding the dataset, labels, vocabulary, tokenizer, sequence length, and training conditions constant.

This helps us understand:

How much these architectures can learn without pretraining

Whether transformers are still strong when built with no pretrained embeddings

Whether the choice of embedding method affects classification performance

# Dataset

We use the Massive Stock News Analysis Database from Kaggle, focusing on the file:

analyst_ratings_processed.csv

We then:

Collected price data for the 10 most frequent tickers:
MRK, MS, MU, NVDA, QQQ, M, EBAY, NFLX, GILD, VZ

Merged each news headline with the price on the same day

Created the next-day label:

UpDownLabel = 1 if NextClose > ClosePrice

Otherwise 0

Performed a strict time-based split:

Train: news before 2019

Test: news from 2019–2020

Cleaned text and filtered only valid rows

This produced:

Training samples: 24,090

Testing samples: 5,794


# Methodology
 ### 1. Building the Dataset pipeline

We implemented:

Full price collection with yfinance

Date alignment between news and prices

Next-day return labeling

Time-based train–test split (to avoid leakage)

Shared tokenizer for both models

Maximum sequence length = 40 tokens

### 2. LSTM With Attention (Scratch-Built)
Architecture

Embedding layer

Bidirectional LSTM (64 units)

Custom Attention Layer

Learns which tokens matter most in the headline

Dense (64) + Dropout (0.3)

Sigmoid output (binary classification)

This produces a learned vector representation (embedding) of each headline that emphasizes important words such as “beats,” “downgrade,” “lawsuit,” “miss,” “upgrade,” etc.

### 3. Transformer (Scratch-Built)

We built a minimal Transformer encoder from scratch, including:

Token embeddings

Positional embeddings

2 encoder layers

Multi-Head Attention (4 heads)

Feed-forward network (128 hidden units)

Layer normalization

Global average pooling

Final classifier

This model also learns embeddings but relies on:

Self-attention across tokens

Positional context learned from the embedding table

Unlike standard BERT, our model has no pretraining, meaning it must learn patterns purely from the headline dataset.

### 4. Training Setup

Both models share:

Sequence length: 40

Embedding dimension: 64

Batch size: 64

Epochs: 5

Optimizer: Adam (1e-3 learning rate)

Loss: Binary cross-entropy

### Results
Performance Summary
Model	Accuracy	F1 Score
LSTM + Attention	0.5005	0.4680
Transformer (Scratch)	0.4803	0.4332
Interpretation

Both models perform better than random guessing (50 percent for accuracy, 0.0 for F1), but our results are inconclusive.

The LSTM consistently outperforms the transformer across metrics

The transformer struggled because scratch-built transformers usually require massive data and pretraining

The LSTM benefits from sequential inductive bias and attention, which helps with short headlines

The results are inconclusive, but show the LSTM has a slight consistent edge

### Conclusions

Scratch-built LSTM with Attention performed slightly better than a scratch-built Transformer

Without pretraining, Transformers lose their usual advantage

Headlines alone (short text) may favor LSTM architectures

Market movement prediction remains extremely challenging due to noise, randomness, and data sparsity

Future work should test pretrained embeddings, such as FinBERT

### Future Work

Use pretrained Transformer embeddings

FinBERT

BERT base

Finance-specific RoBERTa

Test longer text (full articles instead of headlines)

Apply time-series models:

LSTM-attention hybrids

Use more robust prediction targets:

Multi-day movement and shorter window, for example look at the stock price 5 min after the headline went out rather than the day after/


### References

Devlin, J., Chang, M-W., Lee, K., & Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
arXiv:1810.04805.

Araci, D. (2019).
FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
arXiv:1908.10063.

Zeng, Qingyun & Jiang, Tingsong (2025).
Financial Sentiment Analysis Using FinBERT with Application in Predicting Stock Movement.
arXiv:2306.02136v3.

Vaswani, A. et al. (2017).
Attention Is All You Need.
arXiv:1706.03762.

Kaggle — Massive Stock News Database
https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

TensorFlow 2.20.0 Documentation

yfinance Documentation
