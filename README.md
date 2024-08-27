<h1 align="center">
    Image Classification
</h1>

***Fashion MNIST ANN***

This project implements an Artificial Neural Network (ANN) to classify images from the Fashion MNIST dataset. The model is built using TensorFlow and Keras, and it achieves high accuracy in predicting the correct category for each piece of clothing in the dataset.

***Introduction***

The Fashion MNIST dataset is a popular dataset used for benchmarking machine learning models. It contains 70,000 grayscale images in 10 categories. Each image is 28x28 pixels. This project uses an ANN model to classify these images into their respective categories.

***Dataset***

The dataset consists of 10 categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The training set contains 60,000 images, and the test set contains 10,000 images.

***Requirements***

- Python 3.x
- Jupyter Notebook
- Tensorflow
- NumPy
- Matplotlib

***Run the Application***

After installing the necessary libraries, you can run the application using the following command:

```bash
jupyter notebook Fashion_MNIST_ANN.ipynb
```

***Model Architecture***

The model is a simple feedforward neural network with the following architecture:
- **Input Layer**: 784 units (28x28 pixels flattened)
- **Hidden Layers**:
  - 1st Hidden Layer: 128 units, ReLU activation
  - 2nd Hidden Layer: 64 units, ReLU activation
- **Output Layer**: 10 units (one for each class), Softmax activation

The model is trained using the Adam optimizer and categorical crossentropy loss function.

***Results***

The model achieves an accuracy of approximately `89%` on the test set after `20` epochs of training.
