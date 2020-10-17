import torch
import torch.nn as nn 
import torch.nn.functional as F


class LocalCascadeCNN(nn.Module):


    def __init__(self):

        super(LocalCascadeCNN, self).__init__()

        ''' 
        
        One Way Of Implementation 

        self.first_cnn_path_one = nn.Sequential (
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2)
        )

        self.first_cnn_path_two = nn.Sequential (
            nn.Conv2d(in_channels=4, out_channels=160, kernel_size=13)
        )

        self.two_cnn_path_one = nn.Sequential (
            nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2)
        )

        self.two_cnn_path_two = nn.Sequential (
            nn.Conv2d(in_channels=9, out_channels=160, kernel_size=13)
        )

        self.concat_path = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=5, kernel_size=21)
            nn.Softmax(dim=1)
        )

        '''

        # First CNN First Path  
        self.first_cnn_path_one_conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7)
        self.first_cnn_path_one_pool1 = nn.MaxPool2d(kernel_size=4)
        self.first_cnn_path_one_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.first_cnn_path_one_pool2 = nn.MaxPool2d(kernel_size=2)

        # First CNN Second Path
        self.first_cnn_path_two_conv = nn.Conv2d(in_channels=4, out_channels=160, kernel_size=13)

        # First CNN Concat & Output
        self.concat_path1 = nn.Conv2d(in_channels=224, out_channels=5, kernel_size=21)
        self.output1 = nn.Softmax(dim=1)

        # Second CNN First Path
        self.two_cnn_path_two_conv1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7)
        self.two_cnn_path_two_pool1 = nn.MaxPool2d(kernel_size=4)
        self.two_cnn_path_two_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.two_cnn_path_two_pool2 = nn.MaxPool2d(kernel_size=2)

        # Second CNN Second Path
        self.two_cnn_path_two_conv = nn.Conv2d(in_channels=9, out_channels=160, kernel_size=13)

        # Second CNN Concat & Output
        self.concat_path2 = nn.Conv2d(in_channels=224, out_channels=5, kernel_size=21)
        self.output2 = nn.Softmax(dim=1)


    def forward(self, input):
        
        '''

        One Way Of Implementation

        x1 = self.first_cnn_path_one(input)
        
        y1 = self.first_cnn_path_two(input)
        z1 = torch.cat((x1, y1), 0)
        out1 = self.concat_path(z1)

        input_two = torch.cat((x, out1), 1)
        x2 = self.two_cnn_path_one(input_two)
        y2 = self.two_cnn_path_two(input_two)
        z2 = torch.cat((x2, y2), 0)
        out2 = self.concat_path(z2)
        
        return out2

        '''

        # Forward Pass Through First CNN First Path
        x1 = self.first_cnn_path_one_conv1(input)
        x1 = torch.max(x1)
        x1 = self.first_cnn_path_one_pool1(x1)
        x1 = self.first_cnn_path_one_conv2(x1)
        x1 = torch.max(x1)
        x1 = self.first_cnn_path_one_pool2(x1)

        # Forward Pass Through First CNN Second Path
        y1 = self.first_cnn_path_two_conv(x)

        # Concatinating Both The Paths 
        z1 = torch.cat((x1, y1), 0)

        # Forward Pass After Concatinating 
        z1 = self.concat_path1(z1)

        # Output From The First CNN
        out1 = self.output1(z1)

        
        # Forward Pass Through Second CNN First Path
        x2 = self.two_cnn_path_two_conv1(input)
        x2 = torch.max(x2)
        x2 = self.two_cnn_path_two_pool1(x2)

        # Concatinating Output From The First CNN And First Layer Of First Path Of Second CNN
        x2 = torch.cat((out1, x2), 0)

        # Forward Pass Continue With The New Input
        x2 = self.two_cnn_path_two_conv2(x2)
        x2 = torch.max(x2)
        x2 = self.two_cnn_path_two_pool2(x2)

        # Forward Pass Through Second CNN Second Path
        y2 = self.two_cnn_path_two_conv(input)

        # Concatinating Both The Paths 
        z2 = torch.cat((x2, y2), 0)

        # Forward Pass After Concatinating
        z2 = self.concat_path2(z2)

        # Output From The Second CNN
        out2 = self.output2(z2)

        return out2 


model = LocalCascadeCNN()
print(model)