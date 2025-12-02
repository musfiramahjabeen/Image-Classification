# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

To classify the given images into it's category

## Neural Network Model

![423398622-acb92196-8a29-40c9-9505-82bf31af77b3](https://github.com/user-attachments/assets/8b69f643-0ff5-4643-b9ce-f9edd4bdfbdb)

## DESIGN STEPS
### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.

## PROGRAM
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

```python
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
## Train the Model
def train_model(model, train_loader, num_epochs=3):
    # write your code here
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```
## OUTPUT
### Training Loss per Epoch
<img width="252" height="111" alt="image" src="https://github.com/user-attachments/assets/3578a955-7e8e-41e5-bebd-11ab13aefbe4" />



### Confusion Matrix

<img width="709" height="608" alt="image" src="https://github.com/user-attachments/assets/e6898636-9e7b-49d1-a21c-bc516c9b27d3" />



### Classification Report

<img width="613" height="399" alt="image" src="https://github.com/user-attachments/assets/5892182b-adab-4f96-ac41-256ac3456c9b" />




### New Sample Data Prediction

<img width="389" height="432" alt="image" src="https://github.com/user-attachments/assets/cc3795b6-8045-43e1-a30b-e228283c4eee" />



## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
