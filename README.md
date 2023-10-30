# CNN-for-Image-Classification

In this project I will explore how to develop a simple Convolutional Neural Network for image classification. I will use the CIFAR-10 dataset. In the first part, I will show how to develop a simple CNN, while in the second part I will explore the impact of various hyper-parameters on the learning performances.

# Data Loading and Preprocessing

I will use the CIFAR-10 dataset.The dataset consists of 60.000 images in 10 classes, with 6.000 images per class. There are 50.000 training images and 10.000 test images. Each sample is a 32×32 pixels color image (thus with an extra ×3 dimensions for the colors channels), associated with a label from one of the classes:

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
I will divide the dataset in training, testing and validation set. The training set will be used to train the model, the validation set will be used to perform model selection and finally, the test set will be used to asses the performance of deep network.

In the standard dataset, each pixel intensity is represented by a uint8 (byte) from  0  to  255 . As a preprocessing step, we will rescale these values in the range  [0,1] . You should write a simple so-called MinMaxScaler which takes as input a PIL Image (a specific format for images in Python) and rescales it, after making the appropriate type and shape transformations.

# Model Definition and Training

Let's create a simple CNN. The model will be composed of:

One 2D convolutional layer with kernel size 3x3 and 32 output filters/features, that use ReLU activation function
a Max Pooling layer (2D) of size 2x2.
a Flatten layer
a final Dense layer with 10 output neurons (one per class). I do not need to normalize or transform further the outputs, as the CrossEntropyLoss takes care of that. Another equivalent approach would be to add a LogSoftmax final layer that returns log-probabilities and then use the NegativeLogLikelihoodLoss., and with the softmax activation function to ensure that the sum of all the estimated class probabilities for each image is equal to 1.

# Deep CNN

Let's consider a deeper model:

One 2D convolutional layer with kernel size 3x3 and 32 output filters/features, that use ReLu activation function
a Max Pooling layer (2D) of size 2x2
One 2D convolutional layer with kernel size 2x2 and 16 output filters/features, that use ReLu activation function
a Max Pooling layer (2D) of size 2x2
a Flatten layer
a final Dense layer with 10 output neurons (one per class), and with the softmax activation function

I tried different models for the same task.
For the simple CNN accuracy was: 70.92%
For the Deeper model was: 77.58% 
