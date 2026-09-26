### CV Tasks

#### Classic Supervised

* classification
* regression

#### Unsupervised / Self-Supervised

* use-cases
    * good for pretraining a model
    * good to learn representations that can be used in downstream tasks

* methods
    * unsupervised
        * autoencoder
    * self-supervised
        * Natural Language Processing (NLP): Masked Language Modelling
        * Computer Vision: A model might be asked to predict the missing pieces of a scrambled image, guess the correct rotation of a picture, or predict what comes next in a video sequence.
        * Contrastive Learning: A method where the model is shown multiple altered versions (e.g., cropped, recolored, or rotated) of the same image. It learns to recognize that these different versions represent the same object, while treating completely different images as distinct.


#### Semi-Supervised

1. Initial Training: The model is first trained using the small, precious portion of labeled data.
2. Pseudo-Labeling: The partially trained model evaluates the large volume of unlabeled data and makes predictions, effectively assigning its own labels ("pseudo-labels") to the data points it feels confident about.
3. Iterative Refinement: The model is then retrained on the combined dataset of original labeled data and the new pseudo-labeled data, continuously improving its accuracy through feedback loops.


#### Distillation

* Teacher Model
    * A large, cumbersome, pre-trained neural network or ensemble of networks that has already learned to make highly accurate predictions.
* Student Model
    * A significantly smaller, leaner network built to mimic the teacher's outputs, making it ideal for deployment on edge devices or mobile phones.
* Soft Labels vs. Hard Labels
    * Instead of learning only from strict ground-truth dataset labels ("hard targets"), the student learns from the teacher's probability distributions ("soft targets").
* Learning the "Dark Knowledge"
    * Soft labels reveal the relative probabilities of incorrect answers. For example, a model might correctly label an image as a cat, but its soft labels reveal it assigns a 10% probability to dog and a 0.001% probability to car. This extra information teaches the student how the teacher generalizes and what features it looks for.
* Distillation Loss
    * The primary loss that forces the student to match the softened probability distributions (soft labels) of the teacher.
* Student Loss (or Supervised Loss)
    * A secondary loss that ensures the student still learns the ground-truth answers from the original training dataset.


#### Object detection

* Goal: Find, count, and name all the different items of interest in an image.
* Output: Multiple bounding boxes paired with specific class labels (like "car", "dog", or "person") for each box.

* prediction
    * Bounding Box Coordinates (4 values)
        * [$x_{min}$, $y_{min}$, $x_{max}$, $y_{max}$] : top-left, bottom-right corners
        * [$x_c$, $y_c$, $h$, $w$] : center, width, height of bounding box
    * Confidence Score (1 value)
        * The probability that an object actually exists inside that bounding box
    * Class Probabilities (C values)
* loss
    * $L = \lambda_1 L_{bbox} + \lambda_2 L_{conf} + \lambda_3 L_{cls}$
    * $L_{cls} = -\sum_{ic} y_{ic} \log p_{ic}$ - cross-entropy
    * $L_{conf} = BCE$, target is ground truth $IoU$
    * $L_{bbox}$
        * $1 - IoU(\text{predicted bbox}, \text{ground truth bbox})$
        * Complete IoU (CIoU): (1 - IoU) + central point ditance term + aspect ratio term
        * Dice loss $= \frac{2 * intersection}{gt + predicted}$
* multiple object prediction:
    * image is split into grid cells according to feature map
    * make prediction in every grid
* Non-Maximum Suppression
    1. Filter by Confidence: Discard all predicted boxes with a confidence score below a certain threshold (e.g., lower than 25%)
    2. Select the Best Box: For the remaining boxes, pick the one with the absolute highest confidence score and save it as the "final" prediction
    3. Calculate Overlap (IoU): Look at all other neighboring boxes for that same class and calculate their Intersection over Union (IoU)—a percentage score of how much they overlap with the best box.
    4. Suppress Duplicates: If a neighboring box overlaps significantly with the best box (e.g., an IoU greater than 45%), the algorithm assumes it is a duplicate and deletes it.
    5. Repeat: Move to the next highest confidence box and repeat the process until all overlapping duplicates are cleaned up.


#### Localization

* Goal: Find where a single main item lives inside an image.
* Output: Coordinates for one bounding box (box edges, height, and width).
* Use Case: Finding a single face on a passport ID card or locating one tumor in a medical scan.


#### Segmentation

* Semantic
    * Predicts a class label for every pixel, treating all objects of the same class as a single entity (e.g., all "cars" are the same color).
    * The output shape is usually (C, H, W)
    * Cross Entropy Loss
* Instance
    * Instance Segmentation: Detects and delineates each distinct object of interest (e.g., Car 1, Car 2, Car 3). It predicts a class, a bounding box, and a unique mask for every object instance.
    * Training is similair to Semantic segmentation and object detection 


### Modern Architectures

#### LeNet 5 (1998)

* Automatic Feature Extraction: The network learns filters directly from raw pixel data instead of relying on human-engineered edge detection.
* Hierarchical Representation: It processes data in layers—moving from simple, low-level features (like lines) to complex, high-level shapes (like digit contours).
* Weight Sharing: By applying the same filter weights across different parts of the image, the network drastically reduced the number of trainable parameters while maintaining translation invariance.

```py
# https://github.com/lychengrex/LeNet-5-Implementation-Using-Pytorch/tree/master
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,  kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```


#### AlexNet (2012)

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*0dsWFuc0pDmcAmHJUh7wqg.png)

* ReLU Activation: It used the Rectified Linear Unit (\(f(x) = \max(0, x)\)) instead of traditional sigmoid or tanh functions. ReLU prevented vanishing gradients and allowed the network to train much faster.
* Dropout: A regularization technique used in the fully connected layers that randomly "drops out" (turns off) neurons during training. This forced the network to learn robust features and significantly reduced overfitting.
* Overlapping Max-Pooling: Unlike traditional pooling, AlexNet used overlapping pooling windows, which reduced the dimensionality of the data while helping to prevent overfitting.
* Data Augmentation: The team expanded their dataset artificially by applying random translations, horizontal reflections, and altering the RGB channel intensities to improve model generalization.
* Multi-GPU Training: Because GPUs were limited in memory (around 3GB) at the time, the model was split across two GPUs, allowing for parallel computation and faster processing.

```python
# https://github.com/dansuh17/alexnet-pytorch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        return self.classifier(x)
```


#### VGG 16 (2014)

![](https://miro.medium.com/1*_Lg1i7wv1pLpzp2F4MLrvw.png)

* The "16" in VGG16: The model has 16 weight layers: 13 convolutional layers for extracting image features and 3 fully connected layers for classification.
* Uniform Architecture: Every single convolutional layer in the network uses the exact same size filter: 3 × 3 with a stride of 1. Every max-pooling layer uses a 2 × 2 window to downsample the image.
* Receptive Field Innovation: By stacking three 3 × 3 convolutional layers consecutively, the model achieves the same effective "view" of an image as one large 7 × 7 filter. However, using smaller filters requires fewer parameters and introduces three non-linear activation functions instead of one, making the model much better at learning complex features.
* Pioneering Depth: It contains 16 weight layers (13 convolutional layers and 3 fully connected layers). When it was introduced by the Visual Geometry Group (VGG) at Oxford University, this level of depth was a major breakthrough in computer vision.
* Performance: It achieved a 92.7% top-5 test accuracy on the ImageNet dataset, which contains over 14 million images across 1,000 categories.

```python
# https://medium.com/@ilaslanduzgun/create-vgg-from-scratch-in-pytorch-aa194c269b55

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.model(x)
```


#### GoogLeNet / Inception (2014)

![](https://media.licdn.com/dms/image/v2/C4E12AQE6biIwKxVfNA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1601128854140?e=2147483647&v=beta&t=4ceRA4a-KcwLsAmxn0dy5TYsM38cyTlhdF442_y_ntA)

* Parallel Operations: Instead of stacking sequential convolutional layers, an Inception block applies multiple transformations in parallel—typically 1×1, 3×3, and 5×5 convolutions, along with max pooling. The outputs are then concatenated depthwise, allowing the model to choose the most relevant features.
* 1x1 Convolution Bottlenecks: Applying large convolutions (like 5×5) across deep channels is highly resource-intensive. Inception solves this by running 1×1 (pointwise) convolutions to compress the number of input channels before the larger convolutions. This cuts down on computational costs and parameters without sacrificing performance.
* Auxiliary Classifiers: In very deep networks, gradients can diminish before reaching earlier layers. Inception combats this by attaching intermediate branches with classifiers in the middle of the network, which helps inject gradients back during training.
* Inception v2 & v3: Introduced factorization, which breaks down large filters into smaller ones (e.g., a 5×5 convolution is replaced by two consecutive 3×3 convolutions). They also incorporated Batch Normalization, which stabilizes training.

```python
# https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        outinception_3x3_reduced,
        outinception_3x3,
        outinception_5x5_reduced,
        outinception_5x5,
        out_pool
    ):
        super().__init__()

        self.branch1 = ConvBlock(
            in_channels, out_1x1, kernel_size=1, stride=1
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, outinception_3x3_reduced, kernel_size=1),
            ConvBlock(outinception_3x3_reduced, outinception_3x3, kernel_size=3, padding=1),
        )

        # Is in the original googLeNet paper 5x5 conv but in Inception_v2 it has shown to be
        # more efficient if you instead do two 3x3 convs which is what I am doing here!
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, outinception_5x5_reduced, kernel_size=1),
            ConvBlock(outinception_5x5_reduced, outinception_5x5, kernel_size=3, padding=1),
            ConvBlock(outinception_5x5, outinception_5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)

        return torch.cat([y1, y2, y3, y4], 1)


class Inception(nn.Module):
    def __init__(self, img_channel):
        super().__init__()

        self.first_layers = nn.Sequential(
            ConvBlock(img_channel, 192, kernel_size=3, padding=1)
        )

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        out = self.first_layers(x)

        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.max_pool(out)

        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.max_pool(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        return self.fc(out)
```


#### ResNet

![](https://viso.ai/wp-content/uploads/2021/08/residual-neural-network-resnet-resnet50.png)

* Skip Connections (Residual Learning): This is the model's core innovation. ResNet uses "skip connections" (or shortcuts) that jump over a few layers. This allows the original input to pass directly to later layers, solving the vanishing gradient problem (where networks lose the ability to learn as they get too deep).
* Bottleneck Architecture: Instead of standard layers, ResNet-50 uses "bottleneck blocks." Each block consists of three convolutional layers (1 × 1, 3 × 3, and 1 × 1). The first 1 × 1 layer reduces dimensions, the 3 × 3 layer processes the features, and the final 1 × 1 layer restores the original dimension, which drastically reduces computational complexity.

```python
# inspired by https://github.com/FrancescoSaverioZuppichini/ConvNext
import torch
import torch.nn as nn

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, act: type[nn.Module] = nn.ReLU):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,stride=stride, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            act()
        )

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4, stride: int = 1, act: type[nn.Module] = nn.ReLU):
        super().__init__()
        reduced_channels = out_channels // reduction
        self.block = nn.Sequential(
            ConvNormAct(in_channels, reduced_channels, kernel_size=1, act=act),
            ConvNormAct(reduced_channels, reduced_channels, kernel_size=3, stride=stride, act=act),
            ConvNormAct(reduced_channels, out_channels, kernel_size=1, act=act),
        )
        self.connection = nn.Identity() if in_channels == out_channels else ConvNormAct(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.block(pixel_values) + self.connection(pixel_values)
    
class ResNetBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__(
            nn.Identity() if in_channels == out_channels else BottleneckBlock(in_channels, out_channels, stride=2),
            *[BottleneckBlock(out_channels, out_channels) for _ in range(depth if in_channels == out_channels else depth-1)],
        )

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = ConvNormAct(3, 64, kernel_size=7, stride=2)

        self.blocks = nn.Sequential(
            ResNetBlock(64, 64, depth=3),
            ResNetBlock(64, 128, depth=4),
            ResNetBlock(128, 256, depth=6),
            ResNetBlock(256, 512, depth=3),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),

            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.blocks(self.stem(pixel_values)))
```


#### MoblileNet

![](https://miro.medium.com/1*gbKMeCqsuAvfhMEm4MyWAQ.png)

* Depthwise Convolution: Applies a single 3x3 filter to each input channel individually. It maps spatial features but does not combine channels.
* Pointwise Convolution: Applies a 1x1 convolution across all filtered channels. It linearly combines the channel outputs to create entirely new features.
* ReLU6: caps positive activations to a maximum value of 6 to protect precision on low-power mobile hardware

```python
# https://medium.com/@karuneshu21/implement-mobilenet-v1-in-pytorch-fd03a6618321
import torch
from torch import nn
import torch.nn.functional as F

class DepthWiseSeperable(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseSeperable,self).__init__()
        
        # groups used here
        self.depthwise = nn.Conv2d(in_channels=in_channels , out_channels = in_channels , stride = stride , padding = 1, kernel_size = 3 , groups=in_channels , bias = False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , stride = 1 , padding = 0, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self,x):

        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class MobileNetV1(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(MobileNetV1, self).__init__()

        # Initial convolution layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        
        # Depthwise separable convolutions
        self.features = nn.Sequential(
            self.features,
            DepthWiseSeperable(32, 64, 1),
            DepthWiseSeperable(64, 128, 2),
            DepthWiseSeperable(128, 128, 1),
            DepthWiseSeperable(128, 256, 2),
            DepthWiseSeperable(256, 256, 1),
            DepthWiseSeperable(256, 512, 2),
            
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),
            DepthWiseSeperable(512, 512, 1),

            DepthWiseSeperable(512, 1024, 2),
            DepthWiseSeperable(1024, 1024, 1)

        )
        
        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```


#### ConvNeXt

![](https://user-images.githubusercontent.com/29043871/197199637-9cd8b61b-632a-4cad-9361-b2db0af8c574.png)

* Transformer-Inspired Design: ConvNeXt borrows highly successful design principles from the Swin Transformer (like macro-architecture layout) but applies them without using heavy self-attention mechanisms.
* "Patchify" Stem: Instead of using the traditional large, overlapping 7×7 convolution at the very beginning of the network, ConvNeXt uses a non-overlapping 4×4 convolution with stride 4. This mimics the patch-embedding step of Vision Transformers, which improves efficiency.
* Inverted Bottleneck Blocks: Inspired by MobileNetV2, the block design is inverted. The network first expands the number of channels, applies a depthwise convolution, and then projects it back. This perfectly aligns with the dimension-expansion tricks used in transformer feed-forward layers.
* Depthwise Separable Convolutions: ConvNeXt uses depthwise convolutions, a type of grouped convolution where the number of groups equals the number of channels. This separates spatial and channel-wise mixing, drastically reducing computations (FLOPs) and parameters without sacrificing accuracy.
* Layer Normalization (Instead of Batch Normalization): Traditional CNNs rely heavily on Batch Normalization. ConvNeXt swaps this out for Layer Normalization, which is much more stable and aligns perfectly with transformer-style architectures.
* Larger Kernel Sizes: Moving away from standard 3×3 kernels, ConvNeXt uses larger 7×7 convolutional kernels. This increases the effective receptive field (the area of the input image that each pixel in the final layer affects), mimicking the broad, global attention of transformers.
* Updated Activations and Regularization: It replaces the traditional ReLU activation with GELU (Gaussian Error Linear Unit) and utilizes modern training techniques like Mixup, CutMix, and Stochastic Depth to prevent overfitting.

```python
# https://github.com/FrancescoSaverioZuppichini/ConvNext
import torch
import torch.nn as nn

class ConvNextLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape)
    
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class ConvNextEmbeddigns(nn.Sequential):
    def __init__(self, dim: int, patch_size: int = 4):
        super().__init__(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            ConvNextLayerNorm(dim),
        )

class InvertedBottleneckBlock(nn.Module):
    def __init__(self, features, expansion: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=7, stride=1, padding=3, groups=features),
            ConvNextLayerNorm(features),
            nn.Conv2d(features, features * expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(features * expansion, features, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class ConvNextBlock(nn.Sequential):
    def __init__(self, in_features, out_features, depth: int):
        super().__init__(
            nn.Identity() if in_features == out_features else nn.Conv2d(in_features, out_features, kernel_size=2, stride=2),
            *[InvertedBottleneckBlock(out_features) for _ in range(depth)]
        )

class ConveNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = ConvNextEmbeddigns(64)
        
        self.encoder = nn.Sequential(
            ConvNextBlock(64, 64, 3),
            ConvNextBlock(64, 128, 3),
            ConvNextBlock(128, 256, 9),
            ConvNextBlock(256, 512, 3),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),

            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(self.embeddings(pixel_values)))
```


#### Vision Transformer (ViT)

![](https://tintn.github.io/assets/posts/Implementing-Vision-Transformer-from-Scratch/ViT-arch.png)

```python
# inspired by https://medium.com/@bskkim2022/paper-reimplementation-vit-vision-transformer-eed3ad20dfe7
import torch
from torch import nn
import math

class ViTEmbeddings(nn.Module):
    def __init__(self, image_size: int, patch_size: int, hidden_size: int) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(3, hidden_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(hidden_size))
        self.pos_embeddings = nn.Parameter(torch.randn(self.num_patches, hidden_size))
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values.shape = (batch_size, 3, image_size, image_size)
        B = pixel_values.shape[0]
        patch_emb = self.patch_proj(pixel_values) # batch_size, hidden_size, num_patches**0.5, num_patches**0.5
        patch_emb = patch_emb.flatten(2)          # batch_size, hidden_size, num_patches
        patch_emb = patch_emb.permute(0, 2, 1)    # batch_size, num_patches, hidden_size
        patch_emb += self.pos_embeddings
        embeddings = torch.concat([self.cls_token.expand(B, 1, -1), patch_emb], dim=1) # batch_size, 1 + num_patches, hidden_size

        return embeddings # (batch_size, 1 + num_patches, hidden_size)
    
class ViTMultiHeadSelfAttention(nn. Module):
    def __init__(self, num_attention_heads: int, hidden_size: int) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_size = int(hidden_size / num_attention_heads)

        self.query = nn. Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states) -> torch.Tensor:
        #hidden_states.shape = (batch_size, 1 + num_patches, hidden_size)
        B, N, C = hidden_states.shape
        key_layer = self.key(hidden_states).reshape(B, N, self.num_attention_heads, self.head_size).permute(0, 2, 1, 3)     # (batch_size, num_heads, 1 + num_patches, head_size)
        value_layer = self.value(hidden_states).reshape(B, N, self.num_attention_heads, self.head_size).permute(0, 2, 1, 3) # (batch_size, num_heads, 1 + num_patches, head_size)
        query_layer = self.query(hidden_states).reshape(B, N, self.num_attention_heads, self.head_size).permute(0, 2, 1, 3) # (batch_size, num_heads, 1 + num_patches, head_size)

        attn_scores = torch.einsum("bhnd,bhmd->bhnm", query_layer, key_layer) # (batch_size, num_heads, 1 + num_patches, 1 + num_patches)
        attn_scores = attn_scores / math.sqrt(self.head_size)
        attn_scores = nn.functional.softmax(attn_scores, dim=-1)
        
        attn_values = torch.einsum("bhnm,bhmd->bhnd", attn_scores, value_layer).permute(0, 2, 1, 3).reshape(B, N, C)
        attention_output = self.output(attn_values)

        return attention_output # (batch_size, 1 + num_patches, hidden_size)
    
class ViTLayer(nn. Module):
    def __init__(self, num_attention_heads: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mha = ViTMultiHeadSelfAttention(num_attention_heads, hidden_size)
        self.dense = nn.Sequential(nn.Linear(hidden_size, intermediate_size), nn.GELU(), nn.Linear(intermediate_size, hidden_size))

    def forward(self, hidden_states: torch. Tensor) -> torch. Tensor:
        # hidden_states.shape = (batch_size, 1 + num_patches, hidden_size)
        hidden_states = hidden_states + self.mha(self.norm1(hidden_states))
        hidden_states = hidden_states + self.dense(self.norm2(hidden_states))
        return hidden_states # (batch_size, 1 + num_patches, hidden_size)

class ViTModel (nn. Module):
    def __init__(
            self,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            intermediate_size: int = 3072,
            image_size: int = 224,
            patch_size: int = 16,
        ):
        super().__init__()
        self.embed = ViTEmbeddings(image_size, patch_size, hidden_size)
        self.encoder_layers = nn.Sequential(*[ViTLayer(num_attention_heads, hidden_size, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, pixel_values: torch.Tensor) -> torch. Tensor:
        # pixel_values.shape = (batch_size, 3, image_size, image_size)
        x = self.embed(pixel_values)
        x = self.encoder_layers(x)

        return x # (batch_size, 1 + num_patches, hidden_size)
```


#### U-Net

![](https://miro.medium.com/1*f7YOaE4TWubwaFF7Z1fzNw.png)


### Serving And Optimization

* onnx runtime
* vino
* torch script
* torch compile
* mixed precision
* fsdp
    * data parallelism
        * every gpu aggregates gradients for a mini-batch
        * gradients are accumulated
        * step is made
        * every gpu holds full model, gradients, optimizer state
    * tensor parallelism
        * weights are distributed between gpus
        * for example store columns of linear layer on different gpus
        * also distributes gradients, optimizer state
        * when layer weights need to be updated, make request to other gpus
* quantization
    * real = scale * (quantized - zero_point)
    * r_max = s * (q_max - z)
    * r_min = s * (q_min - z)
    * 8 bit: trivial
        * channel-wise
        * group-wise
    * 2 bit:
        * packing
        * unpacking
* nvidia triton
* flash attention
