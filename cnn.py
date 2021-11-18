import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np


model_file = 'model.pth'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
device = "cpu"
model=Net().to(device)
model.load_state_dict(torch.load(model_file))
#print(model)
model.eval()

# Function to predict the class of an image
def predict_image(classifier, image_array):
   
    
    classifier.eval()
        
    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Preprocess the imagees
    image_tensor = transformation(image_array).float() 
    image_tensor=image_tensor.unsqueeze(0)
    # Turn the input into a Variable
    input_features = image_tensor

    # Predict the class of each input image
    predictions = classifier(input_features)
    
     # Convert the predictions to a numpy array 
    for prediction in predictions.data.numpy():
        class_idx = np.argmax(prediction)
        
    return class_idx 





def main(image_arrays):
    predictions = predict_image(model, np.array(image_arrays))
    return predictions


    
    