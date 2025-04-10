import torch
from torch.nn import BCELoss
from torch.optim import Adam

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for text, labels in dataloader:
        text, labels = text.to(device), labels.to(device).float()
        optimizer.zero_grad()
        output = model(text).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (output >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)