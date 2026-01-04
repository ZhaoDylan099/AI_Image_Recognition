import torch
import pandas as pd
import numpy as np
from loadData import load_data
from CNN import CNNModel


    
def train_model(train, model, num_epochs=10, learning_rate = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for i, data in enumerate(train, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)


            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item() * images.size(0)
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        print('Finished Training')


def evaluate_model(val, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in val:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the validation images: {100 * correct / total} %')


if __name__ == "__main__":
    DATA_DIR = "data/images"
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001

    train_load, val_load = load_data(DATA_DIR, batch_size=BATCH_SIZE, img_size=(224, 224))

    print(len(train_load.dataset), len(val_load.dataset))
    model = CNNModel()

    train_model(train_load, model, num_epochs=EPOCHS, learning_rate=LEARNING_RATE)

    evaluate_model(val_load, model)

    print("Training complete.")

    model_path = "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
