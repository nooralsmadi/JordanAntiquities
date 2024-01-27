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
* <b>Inception V3 Model<b> (Running fine-tune-inception-v3 with 90.9%.ipynb):
     * Architecture Overview:
     * Transfer Learning Approach:
       We remove the head layer, and then add our own layer on top. This allows you to leverage the features learned by the pre-trained 
       model and adapt them to a new task or dataset.
       <b>The layers that we added<b>:     
         Global Average Pooling Layer: The `GlobalAveragePooling2D()` layer reduces the spatial dimensions of the features extracted by 
          the base model to a vector of fixed size, effectively summarizing the learned features for each image.

         Dense Layer 1: The `Dense(units=256)` layer is a fully connected layer with 256 units. This layer helps in learning higher- 
             level features from the output of the global average pooling layer.

         Activation (ReLU): The ReLU activation function introduces non-linearity to the network, allowing it to learn complex 
             patterns in the data.

         Batch Normalization: Batch normalization helps stabilize and accelerate the training of deep neural networks by normalizing 
             the input to each layer.

         Dropout: The `Dropout(rate=0.5)` layer randomly sets a fraction of input units to zero during training, which helps prevent 
             overfitting by forcing the network to learn redundant representations.

         Dense Layer 2 (Output Layer): The final `Dense(units=classes)` layer is the output layer with units equal to the number of 
             classes in the classification task.

          Activation (Softmax): Softmax activation converts the raw output scores into probabilities, where each value represents the 
             probability of the input belonging to a particular class.


     * Learning Curve and Performance Metrics:  
* <b>EfficientNet B0 Model<b> (Running fine-tune-efficient-net-b0 with 84%.ipynb):
     * Architecture Overview:
     * Transfer Learning Approach:
     * Learning Curve and Performance Metrics: 
* <b>VGG16 Model<b> (Running fine-tune-vgg16 with 86.7%.ipynb):
     * Architecture Overview:
     * Transfer Learning Approach:
     * Learning Curve and Performance Metrics: 
* <b>ResNet101 Model<b> (Running fine-tune-resnet-101 with 89%.ipynb):
     * Architecture Overview:
     * Transfer Learning Approach:
     * Learning Curve and Performance Metrics: 
  



