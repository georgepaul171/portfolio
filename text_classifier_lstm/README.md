IMDB Sentiment Analysis with LSTM
This repository implements a binary sentiment classifier using an LSTM network on the IMDB movie reviews dataset. It leverages PyTorch and TorchText for data handling, model construction, training, and evaluation.

Overview
The project performs the following steps:

Data Loading and Preprocessing:
Downloads and processes the IMDB dataset using TorchText. Reviews are tokenized using the basic_english tokenizer, and a vocabulary is built from the tokens.

Model Architecture:
Implements an LSTM-based classifier that processes the tokenized input sequences and outputs a probability prediction (0.0 for negative and 1.0 for positive) using a Sigmoid activation.

Training and Evaluation:
Includes training and evaluation loops using Binary Cross Entropy loss, with accuracy computed per epoch.

Requirements
The project requires Python 3.10 and the following packages:

torch==2.2.0

torchtext==0.17.0

matplotlib==3.9.0

numpy==1.26.4

portalocker>=2.0.0 (needed for caching in torchdata)

These dependencies are listed in the requirements.txt file.

Setup
Follow these steps to set up your environment:

Clone the Repository:

bash
Copy
git clone <repository_url>
cd text_classifier_lstm
Create and Activate a Virtual Environment:

bash
Copy
python3.10 -m venv venv
source venv/bin/activate
Install the Dependencies:

bash
Copy
pip install --upgrade pip
pip install -r requirements.txt
Usage
Running the Training
Execute the main script:

bash
Copy
python main.py
First-Time Run:
An active internet connection is required to download the IMDB dataset. After the initial download, the data is cached locally so subsequent runs can work offline.

Training Output:
You will see logs for the sample from the dataset, followed by epoch-wise training and validation loss/accuracy. Debug prints (if not removed) will show the converted labels per batch.

Expected Runtime
On a CPU:
On-the-fly tokenization and data loading can be a bottleneck. Depending on your machine, each epoch might take 1–2 minutes.

On a GPU:
Model computation should be faster; however, data loading might remain the same unless further optimized (e.g., using multiple workers).

Note: Once you verify that the conversion of labels is correct (showing only 0.0 and 1.0), consider removing or commenting out the debug print statements to clean up the output and potentially speed up training.

Project Structure
main.py:
The entry point; sets up device configuration, loads data, initializes the model, and kicks off training.

data.py:
Handles dataset loading, tokenization, vocabulary building, and defines the collate_batch function.

model.py:
Contains the definition of the LSTM-based classification model.

train.py:
Implements the training loop.

utils.py:
Provides the evaluation function.

requirements.txt:
Lists all required packages.

README.md:
This file.

Troubleshooting
Long Training Times:
If training is slow:

Check if you are running on a CPU. GPU acceleration significantly reduces training times.

Consider increasing the num_workers parameter in the DataLoader to speed up data loading.

Disable or remove debug print statements once you verify label conversion.

Data Download:
An internet connection is required during the first run to download the IMDB dataset. Subsequent runs use the locally cached version.

Label Conversion:
Verify the label conversion in data.py. If issues occur with the target values for BCELoss, ensure that the labels are correctly remapped to either 0.0 or 1.0.

License
[Specify License Here, e.g., MIT License] 
