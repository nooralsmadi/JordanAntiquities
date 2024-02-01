# Image Classification for Archaeological Sites using Pre-trained Models

# Overview:
Archaeological research often involves the analysis of images to identify and classify various archaeological sites. This project aims to develop a deep learning model for automated image classification of archaeological sites. Using pre-trained models will be crucial in achieving accurate and efficient results.

# Objectives:

   * Develop a deep-learning model for image classification 
   * Evaluate multiple pre-trained models: InceptionV3, Efficientnet, ResNet, VGGNet 
   * Compare and analyze the performance of each model 
   * Provide a user-friendly demo using Gradio or Streamlit

# Data Collection:
We collect images for archaeological sites (Umm Qais, Jerash, Petra, Ajloun Castle, Wadi Rum, Roman amphitheater)
using:
* Web scraping: we use web scraping in two ways: using giving URL to download images and using giving some keywords, to search about 
  them then download images. 
* Videos: we converted some videos to frames and added some frames to our dataset.  
    
you can find our dataset in the following link: <b>https://drive.google.com/file/d/1aOWaA5UcroibyKtIm0JskpIEPKh-6Fef/view?usp=sharing</b>

| Name               | Number of images |
| ------------------ | --------------- |
| Ajloun             | 681             |
| Jerash             | 560             |
| Petra              | 519             |
| Roman Amphitheater | 535             |
| Umm Qais           | 566             |
| Wadi Rum           | 827             |

The total number of images in our dataset is **3688 images**.

* <b>Sample Images</b>

    ![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/001.jpg)
    ![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/003.jpg)
    ![3](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/010.jpg)
    ![4](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/018.jpg)
    ![5](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/091.jpg)
    ![6](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/012.jpg)


# Preprocessing steps:
In all Models, we do the same general preprocessing steps that are:
* Image pixel rescaling: divide pixels by 255 to convert the pixels range from 0-255 to 0-1.
* Image resizing: this step may vary depending on the model used, as each model might have a compatible target size for optimal performance. Below are the target sizes used for resizing images for different models:

| Model         | Target Size Used |
|---------------|------------------|
| Inception V3  | 229              |
| EfficientNet  | 224              |
| VGG16         | 224              |
| ResNet101     | 224              |


* One-hot encoding using LabelBinarizer.
* Data Augmentation.
  We split the data into training and testing sets with an 80:20 ratio, then we apply data augmentation exclusively to the training data.

  Data augmentation is a technique used to enhance the size and diversity of a training dataset by applying various transformations to existing images. This process is pivotal for improving the generalization and robustness of trained models, as it exposes them to a wider range of input variations.

    #### Parameters:

    * Rotation_range: Random rotation range for images, specified in degrees.
    * Horizontal_flip: Boolean indicating random horizontal flipping of images.
    * Width_shift_range: Random width shifting range (horizontal direction) for images.
    * Height_shift_range: Random height shifting range (vertical direction) for images.
    * Shear_range: Random shear transformation range for images.
    * Zoom_range: Random zooming range for images.
    * Fill_mode: Method for filling newly created pixels introduced by transformations; 'nearest' fills with the nearest neighbor value.
  
After defining the augmentation parameters with ImageDataGenerator, the flow() method generates augmented batches of training data from the input images (X_train) and corresponding labels (y_train). The BATCH_SIZE parameter determines the size of each augmented data batch during training. 

  ![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/Augmentation.JPG)


# Modeling

## Inception V3 Model (Running fine-tune-inception-v3 with 90.9%.ipynb)

Inception v3 is a convolutional neural network (CNN) architecture designed for image classification tasks. It is part of the Inception family of models developed by Google Research. Inception v3 builds upon the success of its predecessors, Inception v1 and Inception v2, by introducing several improvements in terms of accuracy and computational efficiency.

### Key Features:

- Inception Modules: The hallmark of the Inception architecture is its use of inception modules, which are composed of multiple parallel convolutional pathways of different kernel sizes. These pathways allow the model to capture features at various scales and resolutions simultaneously, facilitating better representation learning.
- Factorization: Inception v3 employs factorization techniques such as factorized convolutions and dimensionality reduction to reduce the computational cost of the model while preserving its expressive power. This helps in making the model more efficient and suitable for deployment on resource-constrained devices.
- Auxiliary Classifiers: Inception v3 includes auxiliary classifiers at intermediate layers of the network, which serve two purposes: providing additional regularization during training and facilitating the gradient flow through the network. These auxiliary classifiers help in mitigating the vanishing gradient problem and improving the overall training stability.
- Batch Normalization: Batch normalization is extensively used throughout the network to accelerate training convergence and reduce the sensitivity to initialization. It normalizes the activations of each layer to have zero mean and unit variance, which helps in improving the gradient flow and reducing the internal covariate shift.
- Global Average Pooling: Instead of fully connected layers at the top of the network, Inception v3 replaces them with global average pooling layers. This reduces the number of parameters in the model and makes it more robust to spatial translations and distortions in the input images.

### Architecture Overview:
Inception v3 consists of multiple convolutional layers followed by inception modules and auxiliary classifiers. The overall architecture can be summarized as follows:

- Input Layer: Accepts input images of predefined dimensions.
- Convolutional Layers: A series of convolutional layers with varying filter sizes and depths to extract hierarchical features from the input images.
- Inception Modules: These modules are the building blocks of the network and contain parallel convolutional pathways of different kernel sizes. They enable the network to capture features at multiple scales and resolutions.
- Auxiliary Classifiers: Auxiliary classifiers are inserted at intermediate layers of the network to aid in training and regularization.
- Final Layers: Global average pooling layers followed by fully connected layers and softmax activation for predicting the class probabilities.

![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/inception%20architecture.png)

### Transfer Learning Approach:

In our transfer learning approach, we remove the head layer of the Inception V3 pre-trained model and then add our own layers on top. This allows us to leverage the features learned by the pre-trained model and adapt them to a new task or dataset.

#### The layers that we added:

- Global Average Pooling Layer: The `GlobalAveragePooling2D()` layer reduces the spatial dimensions of the features extracted by the base model to a vector of fixed size, effectively summarizing the learned features for each image.
- Dense Layer 1: The `Dense(units=256)` layer is a fully connected layer with 256 units. This layer helps in learning higher-level features from the output of the global average pooling layer.
- Activation (ReLU): The ReLU activation function introduces non-linearity to the network, allowing it to learn complex patterns in the data.
- Batch Normalization: Batch normalization helps stabilize and accelerate the training of deep neural networks by normalizing the input to each layer.
- Dropout: The `Dropout(rate=0.5)` layer randomly sets a fraction of input units to zero during training, which helps prevent overfitting by forcing the network to learn redundant representations.
- Dense Layer 2 (Output Layer): The final `Dense(units=classes)` layer is the output layer with units equal to the number of classes in the classification task.
- Activation (Softmax): Softmax activation converts the raw output scores into probabilities, where each value represents the probability of the input belonging to a particular class.

### Learning Curve and Performance Metrics: 
#### Learning Curve:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/inception%20LC.JPG)
#### Performance Metrics: 
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CR-Inception.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CM-Inception.JPG)
#### Samples from predictions:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/predicted-Inception.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/predict2-Inception.JPG)

## EfficientNet B0 Model (Running fine-tune-efficient-net-b0 with 81%.ipynb):
EfficientNet B0 is a convolutional neural network (CNN) architecture proposed by Google Research that achieves state-of-the-art performance on image classification tasks while maintaining high computational efficiency. It is part of the EfficientNet family of models, which are designed to balance model size, accuracy, and computational cost through a novel compound scaling method.

### Key Features:

- Compound Scaling: EfficientNet B0 introduces compound scaling, a method that uniformly scales the network depth, width, and resolution to achieve better performance. By jointly scaling these dimensions, the model achieves higher accuracy without significantly increasing computational cost.
- Depthwise Separable Convolutions: The backbone of EfficientNet B0 is composed of depthwise separable convolutions, which factorize standard convolutions into depthwise convolutions and pointwise convolutions. This reduces the number of parameters and computations required, making the model more efficient.
- Inverted Residual Blocks: Inspired by MobileNetV2, EfficientNet B0 incorporates inverted residual blocks, which consist of a lightweight bottleneck layer followed by a linear projection layer. This design enables efficient feature extraction while maintaining representational power.
- Squeeze-and-Excitation (SE) Blocks: To enhance feature discrimination, EfficientNet B0 integrates SE blocks, which dynamically recalibrate channel-wise feature responses based on their importance. This mechanism helps the model focus on informative features and suppress irrelevant ones.
- Swish Activation Function: EfficientNet B0 employs the Swish activation function, a smooth and non-monotonic activation function that has been shown to improve training convergence and model performance compared to traditional activation functions like ReLU.

### Architecture Overview:

EfficientNet B0 follows a hierarchical structure with multiple stages of feature extraction. The overall architecture can be summarized as follows:

- Input Layer: Accepts input images of predefined dimensions.
- Stem Convolutional Layers: Initial convolutional layers that extract low-level features from the input images.
- EfficientNet Blocks: The backbone of the network consists of multiple EfficientNet blocks, each comprising a sequence of inverted residual blocks with varying depths and widths. These blocks progressively extract higher-level features from the input.
- Global Average Pooling: A global average pooling layer aggregates spatial information across feature maps to generate a compact representation of the input.
- Fully Connected Layer: The global average pooling output is fed into a fully connected layer followed by a softmax activation function for predicting class probabilities.

![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/The-EffecientNet-B0%20architecture.png)

### Transfer Learning Approach:
For this custom neural network architecture, we employ transfer learning, leveraging a pre-trained base model's learned features. Here's the breakdown of the approach:

1. Base Model: We start with a pre-trained base model, which has already learned features from a large dataset (such as ImageNet).
2. Custom Fully Connected Layers: On top of the base model, we add custom fully connected layers to adapt the learned features for our specific classification task.

   - Flattening Layer: The Flatten() layer is used to flatten the output of the base model's last convolutional layer into a 1D tensor.  
   - Dense Layer 1: The Dense(units=1024) layer is a fully connected layer with 1024 units, allowing the model to learn high-level features.
   - Activation (ReLU): The ReLU activation function introduces non-linearity to the network, aiding in learning complex patterns.    
   - Dropout: The Dropout(rate=0.5) layer randomly sets a fraction of input units to zero during training to prevent overfitting.    
   - Dense Layer 2 (Output Layer): The final Dense(units=classes) layer is the output layer with units equal to the number of classes in the classification task.   
   - Activation (Softmax): Softmax activation converts the raw output scores into probabilities, indicating the likelihood of each class.

### Learning Curve and Performance Metrics: 
#### Learning Curve:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/efficient.JPG)

#### Performance Metrics:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CR-Effitientnet.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CM-Effitientnet.JPG)

### Samples from prediction: 
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/predicted%20Efficient.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/efficient2.JPG)

## VGG16 Model (Running fine-tune-vgg16 with 88%.ipynb):

VGG16 is a convolutional neural network (CNN) architecture proposed by the Visual Geometry Group (VGG) at the University of Oxford. It was introduced in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Simonyan and Zisserman. VGG16 is known for its simplicity and effectiveness, and it serves as a popular benchmark model for various computer vision tasks, particularly image classification.

### Key Features: 

- Uniform Architecture: VGG16 follows a uniform architecture, consisting of a series of convolutional layers followed by max-pooling layers. This simplicity enables easy implementation and interpretation of the model.
- Stacked Convolutional Layers: The core building blocks of VGG16 are stacks of convolutional layers with small 3x3 filters, which are followed by rectified linear unit (ReLU) activation functions. This design facilitates feature extraction across multiple layers of abstraction.
- Max Pooling Layers: Max-pooling layers are interspersed between the convolutional layers to downsample feature maps and reduce spatial dimensions, while retaining important features.
- Fully Connected Layers: Towards the end of the network, VGG16 includes fully connected layers followed by softmax activation for classifying input images into various categories. These layers aggregate features learned by the convolutional layers and produce final predictions.
- Pre-Trained Models: Pre-trained versions of VGG16, trained on large-scale image datasets such as ImageNet, are widely available. These pre-trained models can be fine-tuned on specific tasks or used as feature extractors for transfer learning.

### Architecture Overview:

VGG16 architecture consists of the following components:

- Input Layer: Accepts input images of fixed size (usually 224x224 pixels).
- Convolutional Blocks: The network comprises several convolutional blocks, each containing multiple convolutional layers followed by max-pooling layers. These blocks progressively extract hierarchical features from the input images.
- Flattening Layer: The output feature maps from the convolutional blocks are flattened into a vector representation to be fed into fully connected layers.
- Fully Connected Layers: The flattened features are passed through fully connected layers, which perform classification based on learned features. The final layer typically uses softmax activation to output class probabilities.

![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/VGG16%20arch.png)

### Transfer Learning Approach:
The `build_network` function constructs a new neural network architecture by adding custom layers on top of a pre-trained model. Here's a breakdown of the transfer learning approach:

1. Base Model: The function takes a pre-trained model as input, leveraging its learned features from a large dataset or task.
2. Custom Fully Connected Layers:
   - Flattening Layer: The Flatten() layer is applied to the output of the pre-trained model's last layer to convert it into a 1D tensor.
   - Dense Layer 1: A Dense layer with 256 units is added, enabling the model to learn higher-level features.
   - Activation (ReLU): The ReLU activation function introduces non-linearity to the network, aiding in learning complex patterns.
   - Batch Normalization: Batch normalization is applied to stabilize and accelerate the training process by normalizing the input to each layer.
   - Dropout: Dropout with a rate of 0.5 is used to randomly deactivate a fraction of neurons during training, preventing overfitting.
   - Dense Layer 2 (Output Layer): The final Dense layer with units equal to the number of classes in the classification task.
   - Activation (Softmax): Softmax activation converts the model's raw output into probabilities, representing the likelihood of each class.

### Learning Curve and Performance Metrics: 
#### Learning Curve:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/vgg16.JPG)
#### Performance metrics:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CR-VGG16.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CM-VGG16.JPG)
### Samples from prediction:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/vgg1.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/vgg2.JPG)

## ResNet101 Model (Running fine-tune-resnet-101 with 90.1%.ipynb):

ResNet-101 is a deep convolutional neural network (CNN) architecture proposed by Kaiming He et al. in the paper "Deep Residual Learning for Image Recognition." It is part of the ResNet (Residual Network) family of models known for their ability to train very deep neural networks effectively. ResNet-101 specifically refers to a ResNet architecture with 101 layers, and it has demonstrated state-of-the-art performance on various computer vision tasks, including image classification and object detection.

### Key Features:

Residual Connections: The key innovation of ResNet-101 is the introduction of residual connections, also known as skip connections or shortcut connections. These connections allow the network to learn residual functions by directly adding the input to the output of convolutional layers. This facilitates the training of very deep networks by alleviating the vanishing gradient problem and enabling the flow of gradients through the network.

- Bottleneck Blocks: ResNet-101 employs bottleneck blocks, which consist of a sequence of convolutional layers with 1x1, 3x3, and 1x1 filters. This design reduces the computational cost while increasing the representational power of the network, allowing for deeper architectures with improved performance.
- Global Average Pooling: Instead of using fully connected layers at the end of the network, ResNet-101 replaces them with global average pooling layers. This pooling operation aggregates spatial information across feature maps and produces a fixed-size representation for each feature map, reducing overfitting and the number of parameters in the model.
- Pre-Activation Residual Units: ResNet-101 adopts pre-activation residual units, where batch normalization and ReLU activation are applied before convolution operations. This architecture modification has been shown to improve training convergence and generalization performance.
  
### Architecture Overview:

ResNet-101 architecture consists of the following components:

- Input Layer: Accepts input images of fixed size (e.g., 224x224 pixels).
- Convolutional Layers: The network begins with a series of convolutional layers followed by batch normalization and ReLU activation functions.
- Residual Blocks: The core building blocks of ResNet-101 are residual blocks, each containing multiple residual units. Each residual unit consists of multiple convolutional layers with skip connections.
- Pooling Layers: Max pooling layers are interspersed between residual blocks to downsample feature maps and reduce spatial dimensions.
- Global Average Pooling: The final feature maps are passed through global average pooling layers to produce a compact representation of the input.
- Output Layer: The output of the global average pooling layer is fed into a fully connected layer followed by softmax activation for predicting class probabilities.

![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/resnet101%20architecture.png)
  
### Transfer Learning Approach:
The `build_network` function constructs a custom neural network architecture by adding fully connected layers on top of a pre-trained base model. Here's a breakdown of the transfer learning approach:

  - Base Model: The function takes a pre-trained base model, which has already learned features from a large dataset or task.
  - Global Average Pooling Layer: The GlobalAveragePooling2D() layer is applied to the output of the base model. This layer reduces the spatial dimensions of the features extracted by the base model to a vector of fixed size, effectively summarizing the learned features for each image.
  -  Dense Layer 1: A Dense layer with 1024 units is added, allowing the model to learn high-level features from the pooled features.
  - Activation (ReLU): The ReLU activation function introduces non-linearity to the network, enabling it to learn complex patterns in the data.
  - Dropout: Dropout with a rate of 0.75 is applied to the output of the ReLU layer. This layer randomly sets a fraction of input units to zero during training, helping prevent overfitting by forcing the network to learn redundant representations.
  - Dense Layer 2 (Output Layer): The final Dense layer with units equal to the number of classes in the classification task.
  - Activation (Softmax): Softmax activation is applied to the output layer, converting the raw output scores into probabilities, where each value represents the probability of the input belonging to a particular class.
    
### Learning Curve and Performance Metrics: 
#### Learning Curve:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/LR-Resnet.JPG)
#### Performance Metrics:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CR-RESNET.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/CM-RESNET.JPG)
#### Samples from prediction:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/predicted%20resnet2.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/resnet%20predicted.JPG)
  
# Results and Discussion:
| Model         | Accuracy (%)       | Precision (%)   | Recall  (%)   |
|---------------|--------------------|-----------------|---------------|
| Inception V3  |   91               |    91           |    91         |
| EfficientNet  |   83               |      81         |     81        |
| VGG16         |     88             |       88        |        88     |
| ResNet101     |     90             |        90       |     90        |

### Accuracy:
* Inception V3 achieves the highest accuracy of 91%, indicating that it correctly classifies 91% of the instances in the dataset.
* ResNet101 closely follows with an accuracy of 90%, demonstrating its effectiveness in the task.
* VGG16 performs slightly lower with an accuracy of 88%, still showing strong performance but trailing behind Inception V3 and ResNet101.
* EfficientNet achieves the lowest accuracy among the models at 83%, indicating that it may not generalize as well to the dataset compared to the other architectures.
  
### Precision:
* Inception V3, VGG16, and ResNet101 all achieve precision scores of 91%, indicating that when they predict a certain class, they are correct 91% of the time.
* EfficientNet has a slightly lower precision of 81%, suggesting that it may have a higher rate of false positives compared to the other models.
  
### Recall:
* Inception V3, VGG16, and ResNet101 achieve recall scores of 91%, indicating that they are able to correctly identify 91% of the instances belonging to a certain class.
* EfficientNet also has a recall of 81%, suggesting that it may miss identifying some instances of certain classes more often compared to the other architectures.
  
Overall, Inception V3 and ResNet101 demonstrate the highest overall performance across all metrics, with Inception V3 having a slight edge in accuracy. VGG16 also performs well, closely following the top models in accuracy, precision, and recall. EfficientNet, while still achieving respectable scores, shows a slightly lower performance compared to the other architectures in this particular evaluation.

# Gradio Demo
you can see our Gradio demo at the following link: <b>https://huggingface.co/spaces/nouralsmadi1/JordanAntiquities<b>

you can upload the image and see a description of it.



# Great Job
