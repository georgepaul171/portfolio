import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def yield_tokens(data_iter, tokenizer):
    for label, line in data_iter:
        yield tokenizer(line)

def convert_label(label):
    """
    Converts a label to a float value between 0.0 and 1.0.
    
    If the label is an integer (e.g. 1 or 2), subtract 1 so that:
      - 1 becomes 0.0 
      - 2 becomes 1.0
    
    If the label is a string, map "neg" to 0.0 and "pos" to 1.0.
    """
    if isinstance(label, int):
        return float(label - 1)
    elif isinstance(label, str):
        mapping = {"neg": 0.0, "pos": 1.0}
        return mapping[label]
    else:
        raise ValueError(f"Unexpected label type: {label}")

def get_data(batch_size=64, max_len=200):
    tokenizer = get_tokenizer('basic_english')
    # Build vocabulary using the training split.
    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    def collate_batch(batch):
        # Unpack the batch into labels and texts.
        labels, texts = zip(*batch)
        # Convert texts to token indices.
        texts = [torch.tensor(vocab(tokenizer(text)), dtype=torch.long) for text in texts]
        # Pad the text sequences so that all are the same length.
        texts = pad_sequence(texts, batch_first=True)
        # Convert labels to 0.0 or 1.0 using the helper function.
        converted_labels = [convert_label(label) for label in labels]
        # (Optional) Debug: print out the converted labels.
        labels_tensor = torch.tensor(converted_labels, dtype=torch.float)
        return texts, labels_tensor

    # Reload iterators explicitly for training and testing.
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    # Debug: print a sample to ensure the format is correct.
    sample = next(iter(train_iter))
    print("Sample from dataset:", sample)
    
    # Create DataLoader objects from the dataset lists.
    train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_dataloader, test_dataloader, len(vocab)
