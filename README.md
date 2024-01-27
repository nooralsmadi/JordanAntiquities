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
  ![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/Augmentation.JPG)


# Modeling

## Inception V3 Model (Running fine-tune-inception-v3 with 90.9%.ipynb)

### Architecture Overview:



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
### Performance Metrics: 

## EfficientNet B0 Model (Running fine-tune-efficient-net-b0 with 84%.ipynb):
### Architecture Overview:

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

### Samples from prediction: 
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/predicted%20Efficient.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/efficient2.JPG)

## VGG16 Model (Running fine-tune-vgg16 with 86.7%.ipynb):
### Architecture Overview:

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

### Samples from prediction:
![1](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/vgg1.JPG)
![2](https://github.com/nooralsmadi/JordanAntiquities/blob/main/Data/vgg2.JPG)
## ResNet101 Model (Running fine-tune-resnet-101 with 90.1%.ipynb):
### Architecture Overview:
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
| Model         | Accuracy         | Precision     | Recall      |
|---------------|------------------|---------------|-------------|
| Inception V3  |                  |               |             |
| EfficientNet  |                  |               |             |
| VGG16         |                  |               |             |
| ResNet101     |                  |               |             |




# Great Job
