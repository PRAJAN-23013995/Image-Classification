# NAME : PRAJAN P
# Reg no : 212223240121
# Convolutional Deep Neural Network for Image Classification
## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

## STEP 1:

Import required libraries such as PyTorch, torchvision, NumPy, and Matplotlib. Load and preprocess the image dataset using transformations.

## STEP 2:

Design and implement a Convolutional Neural Network using convolutional layers, pooling layers, and fully connected layers.

## STEP 3:

Train the CNN model using a suitable loss function and optimizer. Evaluate the model using test data and generate performance metrics.


## PROGRAM

### Name:Yashaswini S
### Register Number:212224220123
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 → 7x7

        x = x.view(x.size(0), -1)                  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
from torch.utils.data import DataLoader, TensorDataset

# Dummy data (just to test output)
images = torch.randn(64, 1, 28, 28)   # 64 fake images
labels = torch.randint(0, 10, (64,)) # 64 fake labels

train_dataset = TensorDataset(images, labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

```python
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name: YASHASWINI S')
        print('Register Number: 212224220123')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


```

## OUTPUT
### Training Loss per Epoch

<img width="685" height="292" alt="exp3 dl" src="https://github.com/user-attachments/assets/73d39d00-cfe3-4c4a-a7de-bd7be3e76a1a" />


### Confusion Matrix

<img width="906" height="760" alt="image" src="https://github.com/user-attachments/assets/9780498a-39e1-4ece-9ea6-18634a02c5aa" />


### Classification Report

<img width="650" height="414" alt="image" src="https://github.com/user-attachments/assets/4b3d89eb-a1bd-4f0c-8b14-dfe9f3451030" />


### New Sample Data Prediction

<img width="703" height="603" alt="image" src="https://github.com/user-attachments/assets/ef6d4028-db88-4049-8f98-461e5cb16210" />
 

## RESULT
Thus, a convolutional neural network for image classification was successfully implemented and verified using an Excel-based dataset
