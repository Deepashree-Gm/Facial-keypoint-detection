## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5,padding = 2)
        self.pool = nn.MaxPool2d(4,4)
        self.drop1 = nn.Dropout(0.4,inplace = False)
        self.conv2 = nn.Conv2d(32,128, kernel_size = 5,padding = 2)
#         self.conv3 = nn.Conv2d(64,128,5)
#         self.conv4 = nn.Conv2d(128,256,5)
        self.fc1only = nn.Linear(128*14*14,1024,bias = True)
#         self.fc1 = nn.Linear(256*10*10,256)

        self.fc2 = nn.Linear(1024,136,bias = True)
#         self.bn1 = nn.BatchNorm2d(11)

        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.elu(self.conv1(x)))
#         print('pool1',x.shape)
        x = self.pool(F.elu(self.conv2(x)))
#         print('pool2',x.shape)
#         x = self.pool(F.relu(self.conv3(x)))
#         print('pool3',x.shape)
#         x = self.pool(F.relu(self.conv4(x)))
#         print('pool4',x.shape)
#         x = self.drop1(x)
        
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1only(x))
        x = F.elu(self.fc2(x))


#         x = F.log_softmax(x,dim=1)

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.log_softmax(x,dim = 1)


        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
