import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


#Author: Morteza Farrokhnejad
# Load datasets with the correct delimiter
red_wine = pd.read_csv('Data/winequality-red.csv', delimiter=";")
white_wine = pd.read_csv('Data/winequality-red.csv', delimiter=";")

# Function to preprocess data
def preprocess_data(data):
    X = data.drop(columns=['quality']).values
    y = data['quality'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess red wine data
X_train_red, X_test_red, y_train_red, y_test_red = preprocess_data(red_wine)
# Preprocess white wine data
X_train_white, X_test_white, y_train_white, y_test_white = preprocess_data(white_wine)

# Convert to PyTorch tensors and reshape to (1, 3, 4) for CNN input
def to_tensor_and_reshape(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 11)  # Reshape to (1, 11) as there are 11 features
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 11)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

X_train_red, X_test_red, y_train_red, y_test_red = to_tensor_and_reshape(X_train_red, X_test_red, y_train_red, y_test_red)
X_train_white, X_test_white, y_train_white, y_test_white = to_tensor_and_reshape(X_train_white, X_test_white, y_train_white, y_test_white)

# Define CNN model
class WineCNN(nn.Module):
    def __init__(self):
        super(WineCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)  # Use Conv1d since data is 1D (1, 11)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 7, 64)  # Adjust based on conv output dimensions
        self.fc2 = nn.Linear(64, 10)  # Assuming quality scores range from 0 to 9
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

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
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, preds)
    return accuracy

# Train and evaluate for both datasets
def run_training(X_train, y_train, X_test, y_test, dataset_name):
    model = WineCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    train_model(model, train_loader, criterion, optimizer)
    
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy on {dataset_name} dataset: {accuracy * 100:.2f}%")

# Run training and evaluation for red wine dataset
run_training(X_train_red, y_train_red, X_test_red, y_test_red, "Red Wine")
# Run training and evaluation for white wine dataset
run_training(X_train_white, y_train_white, X_test_white, y_test_white, "White Wine")