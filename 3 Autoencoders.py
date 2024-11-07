import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
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

# Convert data to PyTorch tensors
def to_tensor(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

X_train_red, X_test_red, y_train_red, y_test_red = to_tensor(X_train_red, X_test_red, y_train_red, y_test_red)
X_train_white, X_test_white, y_train_white, y_test_white = to_tensor(X_train_white, X_test_white, y_train_white, y_test_white)

#Author: Morteza Farrokhnejad
# Define Autoencoder
class WineAutoencoder(nn.Module):
    def __init__(self, input_dim=11, encoding_dim=5):
        super(WineAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define Classifier
class WineClassifier(nn.Module):
    def __init__(self, encoding_dim=5, output_dim=10):
        super(WineClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# Training function for Autoencoder
def train_autoencoder(autoencoder, dataloader, criterion, optimizer, epochs=250):
    autoencoder.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            _, decoded = autoencoder(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Autoencoder Loss: {running_loss/len(dataloader)}")

# Training function for Classifier
def train_classifier(classifier, autoencoder, dataloader, criterion, optimizer, epochs=250):
    classifier.train()
    autoencoder.eval()  # Freeze autoencoder weights
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            encoded, _ = autoencoder(inputs)
            outputs = classifier(encoded)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Classifier Loss: {running_loss/len(dataloader)}")

# Evaluation function
def evaluate_model(classifier, autoencoder, data_loader):
    classifier.eval()
    autoencoder.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            encoded, _ = autoencoder(inputs)
            outputs = classifier(encoded)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Run training and evaluation
def run_pipeline(X_train, y_train, X_test, y_test, dataset_name):
    # Instantiate models
    autoencoder = WineAutoencoder()
    classifier = WineClassifier()
    
    # Loss functions and optimizers
    autoencoder_criterion = nn.MSELoss()
    classifier_criterion = nn.CrossEntropyLoss()
    autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # Train Autoencoder
    print(f"Training Autoencoder on {dataset_name} dataset:")
    train_autoencoder(autoencoder, train_loader, autoencoder_criterion, autoencoder_optimizer)
    
    # Train Classifier with encoded features
    print(f"Training Classifier on {dataset_name} dataset:")
    train_classifier(classifier, autoencoder, train_loader, classifier_criterion, classifier_optimizer)
    
    # Evaluate model
    print(f"Evaluating on {dataset_name} dataset:")
    evaluate_model(classifier, autoencoder, test_loader)

# Run pipeline for red wine dataset
run_pipeline(X_train_red, y_train_red, X_test_red, y_test_red, "Red Wine")

# Run pipeline for white wine dataset
run_pipeline(X_train_white, y_train_white, X_test_white, y_test_white, "White Wine")