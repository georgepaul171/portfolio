# IMDB Sentiment Analysis with LSTM

**Description:**  
This repository implements a binary sentiment classifier using an LSTM network on the IMDB movie reviews dataset. It leverages PyTorch and TorchText for data handling, model creation, training, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Overview

This project performs the following steps:

- **Data Loading and Preprocessing:**  
  - Downloads the IMDB dataset (if not already cached) using TorchText.
  - Tokenizes reviews using the `basic_english` tokenizer.
  - Builds a vocabulary from the tokenized data.
  - Collates data into padded batches.

- **LSTM Model Architecture:**  
  - Uses a word embedding layer to convert tokens into dense vectors.
  - Processes the sequences through an LSTM layer.
  - Outputs a single probability using a fully-connected layer followed by a Sigmoid activation.

- **Training and Evaluation:**  
  - Trains the model using Binary Cross Entropy (BCE) loss.
  - Evaluates performance per epoch based on loss and accuracy metrics.

## Requirements

This project requires **Python 3.10** along with the following packages:

- **torch==2.2.0**
- **torchtext==0.17.0**
- **matplotlib==3.9.0**
- **numpy==1.26.4**
- **portalocker>=2.0.0** (Required by `torchdata` for caching)

All dependencies are listed in the [requirements.txt](./requirements.txt) file.

## Setup

Follow these steps to set up the project locally:


1. **Clone the Repository:**

   ```bash
   git clone <YOUR_REPO_URL>
   cd text_classifier_lstm


2. **Create and Activate a Virtual Environment:**

python3.10 -m venv venv
source venv/bin/activate

3. **Install Dependencies:**

pip install --upgrade pip
pip install -r requirements.txt


4. **Usage**
To run the training script, execute:

python main.py

  **First-Time Run:**
You will need an active internet connection as the IMDB dataset will be downloaded and cached locally.

  **Training Output:**
The console will display epoch-wise training and validation loss/accuracy. Debug print statements (such as those for displaying converted labels) may appear for each batch—these can be removed once verification is complete.


**Project Structure**
main.py:
Entry point; orchestrates data loading, model training, and evaluation.

data.py:
Handles dataset loading, tokenization, vocabulary construction, and batch collation.

model.py:
Defines the LSTM-based classifier model.

train.py:
Contains the training loop.

utils.py:
Provides functions for model evaluation.

requirements.txt:
Lists all project dependencies.

README.md:
This file.


**License**
MIT
