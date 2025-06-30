TASK3--IMAGE-CLASSIFICATION-MODEL

COMPANY : CODTECH IT SOLUTIONS

NAME : SIMRAN

INTERN ID : CT04DF1484

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS 

MENTOR : NEELA SANTHOSH 

DESCRIPTION : In Task 3 of the CodTech Machine Learning Internship, we developed an image classification model using Convolutional Neural Networks (CNNs) with the help of the TensorFlow and Keras libraries. The goal of this task was to build a deep learning model capable of recognizing images and classifying them into their respective categories, thereby gaining practical experience in computer vision and deep learning. CNNs are a class of deep neural networks that are particularly effective for analyzing visual imagery. They are widely used in tasks such as image classification, object detection, and image segmentation. In this task, we applied a CNN to the CIFAR-10 dataset, a well-known benchmark dataset for image classification problems. The CIFAR-10 dataset consists of 60,000 color images of 32x32 pixels, categorized into 10 different classes, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.The project began with importing necessary libraries including TensorFlow, Keras, NumPy, and Matplotlib. We then loaded and preprocessed the CIFAR-10 dataset using tf.keras.datasets.cifar10. The dataset was split into training and testing sets, and all pixel values were normalized to the range [0, 1] for faster and more efficient training.We designed a simple yet effective CNN model using Keras’ Sequential API. The architecture of the model consisted of multiple layers:Convolutional layers: To extract spatial features from the input images using filters , MaxPooling layers: To reduce spatial dimensions and control overfitting ,Flatten layer: To convert the 2D feature maps into a 1D vector ,Dense layers: To perform the classification based on the extracted features ,Dropout layer: To reduce overfitting by randomly disabling some neurons during training.The model was compiled with the categorical_crossentropy loss function, the Adam optimizer, and accuracy as the evaluation metric. We then trained the model on the training dataset over multiple epochs, with the validation data being used to monitor performance during training.After training, we evaluated the model on the test set to calculate the final accuracy. The model performed reasonably well given the simplicity of the architecture and the complexity of the CIFAR-10 dataset. We also plotted the training and validation accuracy and loss across epochs to visualize the learning process and to ensure that the model was not overfitting or underfitting.Additionally, a few predictions were visualized to compare the model’s output with actual labels, giving an intuitive sense of how the model was performing.This task provided significant exposure to deep learning, especially in the context of image data. We learned:How to load and preprocess image datasets , How to build and train a CNN using TensorFlow/Keras , How to evaluate and interpret deep learning models , The importance of tuning hyperparameters and model layers.Overall, Task 3 offered a practical and hands-on understanding of CNNs and how they are applied in real-world image recognition problems, making it one of the most important tasks in the internship.

OUTPUT : ACCURACY 
