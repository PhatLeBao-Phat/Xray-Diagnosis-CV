import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import AlexNet_Weights
# from pretrained_net import google_model

class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4),
            torch.nn.Dropout(p=0.5, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            torch.nn.Dropout(p=0.25, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.125, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.Linear(256, n_classes)
        )


    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        # print(x.size)
        x = self.linear_layers(x)
        return x

class Ori_AlexNet(nn.Module):
    def __init__(self, n_classes) -> None:
        super(Ori_AlexNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding='valid'),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2, 0.00002, 0.75, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(96, 256, stride=1, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2, 0.00002, 0.75, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

        )


        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            
            nn.Linear(1000, n_classes),
            # nn.Softmax(), # Maybe not
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(AlexNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # Defining another 2D convolution layer
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # Defining another 2D convolution layer
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # for layer in self.cnn_layers:
        #     if isinstance(layer, nn.Conv2d):
        #         init.kaiming_uniform_(layer.weight)

        self.linear_layers = nn.Sequential(
            # Defining a linear layer
            nn.Dropout(0.5),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            # Defining a linear layer
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            # Defining a linear layer
            nn.Linear(2048, n_classes)
        )

        # for layer in self.linear_layers:
        #     if isinstance(layer, nn.Linear):
        #         init.kaiming_uniform_(layer.weight)

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x

    
class PreTrainedAlexNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(PreTrainedAlexNet, self).__init__()

        self.alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # for params in self.alexnet.parameters():
        #     params.requires_grad = False

        # modify last layer to classification of n_classes classes
        num_features = self.alexnet.classifier[6].in_features
        self.alexnet.classifier[6] = nn.Linear(num_features, n_classes)

        # modify the first layer to accept 1 channel instead of 3
        # This also clears the weight and bias of this layer
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)

        print(self.alexnet)

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run input through convolutional layers
        x = self.alexnet.features(x)
        # run input through pool layer
        x = self.alexnet.avgpool(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.alexnet.classifier(x)
        return x

# GoogLeNet
class ConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts, k, s, p):
        super(ConvBlock, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts, kernel_size=(k, k), stride=(s, s), padding=(p, p)),
            nn.ReLU()
        )

    def forward(self, input_img):
        x = self.convolution(input_img)

        return x

class ReduceConvBlock(nn.Module):
    def __init__(self, in_fts, out_fts_1, out_fts_2, k, p):
        super(ReduceConvBlock, self).__init__()
        self.redConv = nn.Sequential(
            nn.Conv2d(in_channels=in_fts, out_channels=out_fts_1, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_fts_1, out_channels=out_fts_2, kernel_size=(k, k), stride=(1, 1), padding=(p, p)),
            nn.ReLU()
        )
    
    def forward(self, input_img):
        x = self.redConv(input_img)

        return x


class AuxClassifier(nn.Module):
    def __init__(self, in_fts, n_classes):
        super(AuxClassifier, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = nn.Conv2d(in_channels=in_fts, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 128, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.avgpool(input_img)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class InceptionModule(nn.Module):
    def __init__(self, curr_in_fts, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool_proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvBlock(curr_in_fts, f_1x1, 1, 1, 0)
        self.conv2 = ReduceConvBlock(curr_in_fts, f_3x3_r, f_3x3, 3, 1)
        self.conv3 = ReduceConvBlock(curr_in_fts, f_5x5_r, f_5x5, 5, 2)

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(in_channels=curr_in_fts, out_channels=f_pool_proj, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def forward(self, input_img):
        out1 = self.conv1(input_img)
        out2 = self.conv2(input_img)
        out3 = self.conv3(input_img)
        out4 = self.pool_proj(input_img)

        x = torch.cat([out1, out2, out3, out4], dim=1)

        return x

class GoogLeNet(nn.Module):
    def __init__(self, in_fts=1, n_classes=6):
        super(GoogLeNet, self).__init__()
        self.conv1 = ConvBlock(in_fts, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 192, 3, 1, 1)
        )

        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.aux_classifier1 = AuxClassifier(512, n_classes)
        self.aux_classifier2 = AuxClassifier(528, n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024 * 7 * 7, n_classes)
        )

    def forward(self, input_img):
        N = input_img.shape[0]
        x = self.conv1(input_img)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out2 = self.aux_classifier2(x)
        x = self.inception_4e(x)
        x = self.maxpool1(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        # if self.training == True:
        #     return [x, out1, out2]
        # else:
        return x


# Dict of model 
MODEL = {
    'Net' : Net,
    'AlexNet' : AlexNet,
    'Ori_AlexNet' : Ori_AlexNet,
    'TrainedAlexNet': PreTrainedAlexNet,
    'GoogLeNet' : GoogLeNet
}