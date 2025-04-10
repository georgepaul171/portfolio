import torch
from data import get_data
from model import LSTMClassifier
from train import train
from utils import evaluate

def main():
    print("Starting training...")  # Add this line

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, vocab_size = get_data()
    model = LSTMClassifier(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(5):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()