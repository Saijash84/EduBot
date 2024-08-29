import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only errors are logged

# Load a pre-trained ResNet50 model
model_resnet50 = ResNet50(weights='imagenet')
model_resnet50.save('imagenet_model.h5')
# Example in TensorFlow/Keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.save('mnist_model.h5')


# Example in TensorFlow/Keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
# Save the model to a file
model.save('cifar10_model.h5')  # Saves the entire model including architecture, weights, and optimizer state


from tensorflow.keras import layers, models
# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Flattening the 28x28 images into 1D arrays
    layers.Dense(128, activation='relu'),    # Fully connected layer with 128 neurons and ReLU activation
    layers.Dense(10, activation='softmax')   # Output layer with 10 neurons for 10 classes, softmax for classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
model.save('fashion_mnist_model.h5')
# Load the saved model
loaded_model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Evaluate the model on the test data
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')




