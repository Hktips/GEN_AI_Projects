import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train,x_test=x_train/250.0,x_test/250
model=models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train,y_train,epochs=5)
test_loss,test_accuracy=model.evaluate(x_test,y_test)
print(f"Test accuracy{test_accuracy*100:.2f}%")

predictions=model.predict(x_test)
plt.imshow(x_test[0],cmap='gray')
plt.title(f"true value:{y_test[0]},Predicted value:{np.argmax(predictions[0])}")
plt.show()
