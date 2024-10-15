import torch

def train_shaplet_model(model, train_loader, optimizer, epochs = 100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for time_series, labels in train_loader:
            time_series, labels = time_series.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(time_series)
            loss = model.loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader)}')
        