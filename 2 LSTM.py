import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data as before
red_wine = pd.read_csv('Data/winequality-red.csv', delimiter=";")
white_wine = pd.read_csv('Data/winequality-white.csv', delimiter=";")

def preprocess_data(data):
    X = data.drop(columns=['quality']).values
    y = data['quality'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_red, X_test_red, y_train_red, y_test_red = preprocess_data(red_wine)
X_train_white, X_test_white, y_train_white, y_test_white = preprocess_data(white_wine)

#Author: Morteza Farrokhnejad
# Convert data to PyTorch tensors and reshape for LSTM
def to_tensor_and_reshape(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 11, 1)  # (batch, seq_len, input_dim)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 11, 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

X_train_red, X_test_red, y_train_red, y_test_red = to_tensor_and_reshape(X_train_red, X_test_red, y_train_red, y_test_red)
X_train_white, X_test_white, y_train_white, y_test_white = to_tensor_and_reshape(X_train_white, X_test_white, y_train_white, y_test_white)

# Define LSTM model
class WineLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=10):
        super(WineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM returns output and hidden states
        out = out[:, -1, :]  # Take the last output of the sequence for classification
        out = self.relu(self.fc(out))
        return out

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=250):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Run training and evaluation for red wine dataset
def run_pipeline(X_train, y_train, X_test, y_test, dataset_name):
    # Define model, loss, and optimizer
    model = WineLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # Train model
    print(f"Training on {dataset_name} dataset:")
    train_model(model, train_loader, criterion, optimizer)
    
    # Evaluate model
    print(f"Evaluating on {dataset_name} dataset:")
    evaluate_model(model, test_loader)

# Run pipeline for red wine dataset
run_pipeline(X_train_red, y_train_red, X_test_red, y_test_red, "Red Wine")

# Run pipeline for white wine dataset
run_pipeline(X_train_white, y_train_white, X_test_white, y_test_white, "White Wine")