import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# train_images = train_images / 255.0 #grayscale image
# train_labels = train_labels / 255.0

# plt.imshow(train_images[3])

# print()
# print(train_labels[3])


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer
    keras.layers.Dense(128, activation="relu"),  # hidden layer
    keras.layers.Dense(10, activation="softmax")  # output layer
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=10)  # train

test_loss, test_acc = model.evaluate(test_images, test_labels)

print()
print("Tested Acc", test_acc)

prediction = model.predict(test_images)
# prediction = model.predict([test_images[7]]) # predict 7th test image


for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + (str)(test_labels[i]))  # show in GUI plot
    plt.title("Prediction: " + (str)(np.argmax(prediction[i])))
    plt.show()
    # print("Predicted: " + (str)(np.argmax(prediction[i])))
    # print("Actual: " + (str)(test_labels[i]))
    # print()

# print(np.argmax(prediction[0]))  # prediction
