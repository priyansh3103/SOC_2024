
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
from keras import regularizers
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train=x_train.reshape((60000, 28, 28, 1)).astype('float32')
x_test=x_test.reshape((10000, 28, 28, 1)).astype('float32')
x_train=x_train/255
x_test=x_test/255
y_train_encoded = to_categorical(y_train, num_classes=10)
print(y_train_encoded.shape)
model = Sequential([
    Dense(32, input_shape=(28, 28, 1), activation = "relu"),
    Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2,2), strides=2),
    Flatten(),
    Dense(10, activation = "softmax")
])
model.compile(Adam(learning_rate=.005), loss='categorical_crossentropy', metrics=['accuracy', 'precision','Recall','F1Score'])
model.fit(x_train, y_train_encoded, validation_split=0.10, batch_size=128, epochs=4, shuffle = True, verbose=1)
y_test_encoded = to_categorical(y_test)
#Predict
y_prediction = model.predict(x_test)
y_prediction = np.argmax (y_prediction, axis = 1)
y_test_encoded = np.argmax(y_test_encoded, axis=1)
#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(y_test_encoded, y_prediction)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
classes = [0,1,2,3,4,5,6,7,8,9]
sns.heatmap(result, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()








