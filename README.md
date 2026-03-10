
# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Include the Problem Statement and Dataset.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### **Step 1: Define the Objective**

Set the goal of creating a **Convolutional Neural Network (CNN)** capable of recognizing and categorizing handwritten digits from 0 through 9.

### **Step 2: Gather the Dataset**

Work with the MNIST dataset, which provides 60,000 images for training and 10,000 images for testing handwritten digit recognition models.

### **Step 3: Prepare the Data**

Transform the images into tensor format, scale pixel values to a normalized range to improve learning performance, and organize the data into DataLoaders to enable efficient batch-based processing.

### **Step 4: Build the Network Architecture**

Construct a CNN model that includes convolutional layers for extracting features, nonlinear activation functions such as ReLU, pooling layers to reduce spatial dimensions, and fully connected layers to perform final classification.

### **Step 5: Train the Network**

Train the model over several epochs using a suitable loss function like CrossEntropyLoss and an optimization algorithm such as Adam to adjust the model’s parameters.

### **Step 6: Assess Model Performance**

Evaluate the trained model using the test dataset. Calculate accuracy and examine performance further through a confusion matrix and a classification report.

### **Step 7: Save and Deploy the Model**

Store the trained model for later use, display prediction results for visualization, and integrate the model into an application if deployment is required.



## PROGRAM

### Name: PRAJAN P
### Register Number: 212223240121
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # Calculate the input size after pooling
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 output classes for MNIST

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output for fully connected layers
        x = x.view(-1, 32 * 7 * 7) # Reshape the tensor

        # Forward pass through fully connected layers
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train() # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                images = images.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: MOHAMED RASHITH S')
        print('Register Number: 212223243003')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT
### Training Loss per Epoch

<img width="323" height="206" alt="image" src="https://github.com/user-attachments/assets/38490530-8334-4efe-be96-337b44fc1cf0" />


### Confusion Matrix

<img width="910" height="759" alt="image" src="https://github.com/user-attachments/assets/f8af0b17-65e0-4332-a5b3-6f2d1cf2136e" />


### Classification Report

<img width="784" height="489" alt="image" src="https://github.com/user-attachments/assets/64161f2f-563a-4d4a-a9d7-c287e8737880" />


### New Sample Data Prediction


<img width="557" height="613" alt="image" src="https://github.com/user-attachments/assets/8231f31b-3a3e-4078-b4aa-7801f1deb52c" />


## RESULT
The Convolutional Neural Network was successfully implemented for FashionMNIST image classification. The model achieved good accuracy on the test dataset and produced reliable predictions for new images, proving its effectiveness in extracting spatial features from images.
