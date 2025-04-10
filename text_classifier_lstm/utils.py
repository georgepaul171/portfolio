import torch 

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for text, labels in dataloader:
            text, labels = text.to(device), labels.to(device).float()
            output = model(text).squeeze(1)
            loss = criterion(output, labels)
            total_loss += loss.item()
            preds = (output >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)