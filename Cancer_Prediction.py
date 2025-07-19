import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers,models
data=load_breast_cancer()
x,y=data.data,data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#build model
model=models.Sequential([
    layers.Dense(64,activation='relu',input_shape=(x_train.shape[1],)),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

#compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#train model
history=model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))

test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test accuracy:{test_acc*100:.2f}%")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='train Accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('model Accuracy')


plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('model loss')

plt.show()




