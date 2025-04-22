# Major-Project
# Project Title : Parkinson's Disease Detection Using Deep Learning
This project aims to develop a Deep Learning model that analyzes spiral and wave drawings to detect Parkinson’s Disease by identifying subtle motor impairments such as tremors and irregularities.
## Tech Stack
- Backend: Python, Flask
- Frontend: Html, CSS, Javascript
- Models: CNN, DenseNet201, ResNet101
## Prerequisites
- Python 3.11 or higher
- Flask
- Anaconda
- Jupyter notebook
## Dataset
The information collected from Kaggle data repository, which was shared with the public by Zham P.The RMIT University of Melbourne, Australia, was responsible for carrying out the process of data collection. 55 participants were recruited to undergo the process of data collection consisting of 27 Parkinson Group and 28 Non-PD group. Various tests were administered by the researchers to evaluate each participant.They utilized a readily available tablet to sketch the spiral pictures, setting a sheet of paper on it and using an ink pen to sketch. They utilized the same procedure to sketch the wave pictures. They achieved a set of images in the data collection process are presented in below Figures
Click here to download dataset [here](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)

## Step by step procedure for model development 
1.Problem Defination:
    Identify the goal: Detect Parkinson's Disease from spiral and wave images using deep learning.
    
2.Data Collection:
    Collect and organize spiral and wave drawings of healthy individuals and Parkinson’s patients from a publicly available dataset.
    
3.Data Augmentation:
    Apply techniques like rotation, zoom, shear, flipping, and noise addition to increase dataset diversity and improve model generalization.
    
4.Data Preprocessing:
    Resize images to a consistent input size
    Convert to grayscale if necessary
    Normalize pixel values
    
5.Model Building:
    Use Convolutional Neural Networks (CNN), DenseNet201, and ResNet101 to extract features and classify images.
    
6.Model Compilation:
    Set loss function (categorical_crossentropy).
    Use Adam optimizer and track accuracy during training.
    
7.Model Training:
    Train models on the training dataset and validate using a separate validation set. Track accuracy and loss over epochs.
    
8.Model Evaluation:
    Evaluate model performance on the test set using metrics like accuracy, precision, recall, and confusion matrix.
    
9.Model Saving:
    Save the best performing model using model.save('parkinsons_model.h5').
    
10.Deployment:
    Develop a web interface using Flask to upload spiral/wave images and predict Parkinson’s status.
    
## Usage
1. **Run the Flask Application**
   ```bash
   flask run
   ```
2. **Access the webpage**
   Open the browser and go to `http://127.0.0.1:5000` to use the web application
   
## Project Ouput Images
   
