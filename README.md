# Artificial Neural Networks 

## Index

1. [(MLP) MultiLayer Perceptron](#1-mlp-multilayer-perceptron);
2. [(SOM) Self-Organizing Map & (RNN) Recurrent Neural Network](#2-som-self-organizing-map--rnn-recurrent-neural-network);
3. [(SVM) Support Vector Machine & (CNN) Convolutional Neural Network](#3-svm-support-vector-machine--cnn-convolutional-neural-network);

## 1 (MLP) MultiLayer Perceptron

In the MLP notebook, we implemented a MultiLayer Perceptron using the `iris` dataset from `sklearn.datasets`. The Iris dataset is a classic dataset in machine learning, consisting of 150 samples of iris flowers with four features each. The notebook includes data preprocessing, model architecture definition, training, and evaluation of the model's performance.

## 2 (SOM) Self-Organizing Map & (RNN) Recurrent Neural Network
The dataset used in this notebook is the Diabetes dataset from the Scikit-learn repository. This dataset contains 442 instances with 10 features each, and the target is a quantitative measure of disease progression one year after baseline.

### (SOM) Self-Organizing Maps:
The MiniSom class from the minisom library is used to create and train a SOM. Then the target values are associated with the winning neurons in the SOM. Finally a grid search is performed to find the best hyperparameters for the SOM.

### (RNN) Recurrent Neural Network:
A Sequential model is created using TensorFlow and Keras. It includes a SimpleRNN layer and a Dense output layer. Then it was compiled with loss and metrics.

## 3 (SVM) Support Vector Machine & (CNN) Convolutional Neural Network
### (SVM) Support Vector Machine
Once again the dataset used in this notebook, but only for the SVM models, was the Diabetes dataset from the Scikit-learn repository. It contains 442 instances with 10 features each, and the target is a quantitative measure of disease progression one year after baseline.

Since the target is continuous, the most adequate Suppor Vector Machine model is the SV Regressor. However both the **SV Regressor** and the **SV Classifier** were trained with the aid of grid searches due to the academic purposes of comparing their respective performances.

### (CNN) Convolutional Neural Network
While for the CNN models we have used the CIFAR10 dataset, which is a widely used dataset for machine learning and computer vision research and consists of 60,000 32x32 coloured images in 10 different classes, with 6,000 images per class.
