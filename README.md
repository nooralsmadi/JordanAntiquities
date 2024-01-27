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
  



