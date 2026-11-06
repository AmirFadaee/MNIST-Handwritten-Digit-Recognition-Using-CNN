# %%
import numpy as np

# %%
import pandas as pd

# %%
from keras.datasets import mnist

# %%
# x_train        # y_train           # x_test       # y_test
# (train_images , train_labels) , (test_images , test_labels) = mnist.load_data()
# اینترنت تخمیه و باید با numpy فایل فشرده ی .npz رو بخونیم

# %%
df = np.load('mnist.npz')
df

# %%
train_images = df['x_train']
train_labels = df['y_train']
test_images = df['x_test']
test_labels = df['y_test']

# %%
train_images

# %%
train_labels

# %%
test_images

# %%
test_labels

# %%
import matplotlib.pyplot as plt

# %%
train_images[0]

# %%
plt.imshow(train_images[65],cmap='gray')
plt.show()
# داده ی نویز
train_labels[-1]

# %%
train_images[3]
# چون نورون های ما در شبکه عصبی وزن و بایاس دارند و وزن مقدار مهم بودن نورون رو تشکیل میده پس برای مدل ما خوب نیست که در عکس های سیاه سفید تفاوت زیادی بین عدد ها باشه 
# 0 = سیاه
# 255 = سفید
# پس باید عکس ها را scale کنیم

# %%
#MinMaxScaler: (x - min) / (max - min)

train_images = train_images / 255   # (train_images - 0) / (255 - 0)
test_images = test_images / 255     # (test_images - 0) / (255 - 0)

# %%
train_images[3]

# %%
plt.imshow(train_images[0],cmap='gray')
plt.show()

# %%
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPooling2D , Flatten

# %%
CNN = Sequential()

CNN.add(Conv2D(filters=8,kernel_size=(3,3),activation='relu',input_shape=(28,28)))
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Flatten())
CNN.add(Dense(units=32,activation='relu'))
CNN.add(Dense(units=10,activation='softmax')) # در لایه ی آخر به تعداد کلاس های وجود داشته برای جواب باید units بزاریم.
# در Regression باید لایه ی آخر را units = 1

# %%
CNN =Sequential()

CNN.add(Conv2D(filters=8,kernel_size=(3,3),activation='relu',input_shape=(28,28,1))) # (number of images , width , height , number of color channels) (n , w , h , c)
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Flatten())
CNN.add(Dense(units=32,activation='relu'))
CNN.add(Dense(units=10,activation='softmax'))

# %%
CNN.summary()

# %%
CNN.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
hCNN = CNN.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels))

# 60000 = 32 bit * 1875
# چون ما batch رو نگذاشتیم به صورت بای دیفالت روی 32 است.

# %%
plt.plot(hCNN.history['accuracy'],label='train')
plt.plot(hCNN.history['val_accuracy'],label = 'val')
plt.legend()
plt.show()

# %%
outCNN = CNN.predict(test_images)

# %%
outCNN[0]

#عدد علمی
# 2514 => 2.514 e+03
# 0.314 => 3.14 e-1

# %%
plt.imshow(test_images[0],cmap='gray')
plt.show()

# %%
maximum = outCNN[0][0]

i=0
index = 0
for ehtemal in outCNN[0]:
    i = i + 1
    if ehtemal > maximum:
        maximum = ehtemal

        index = i - 1

print(index)

# %%
CNN.save('ocrCNN.h5')

# %%
import keras

# %%
CNN = keras.models.load_model('ocrCNN.h5')
CNN

# %%
from PIL  import Image , ImageOps

# %%
img = Image.open('number.png').convert('L')
img = ImageOps.fit(img,(28,28),Image.Resampling.LANCZOS)
img

# %%
imgArr = np.array(img).astype('float32') / 255
imgArr

# %%
plt.imshow(imgArr , cmap='gray')
plt.show()

# %%
imgArr = 1 - imgArr
imgArr 

# %%
plt.imshow(imgArr,cmap='gray')
plt.show()

# %%
imgArr.shape

# %%
img4D = imgArr.reshape((1,28,28,1)) #( n, w, h, c )
img4D

# %%
img4D.shape

# %%
pishbini = CNN.predict(img4D)
pishbini

# %%
i = 0
index = 0
maximum = pishbini[0][0]

for ehtemal in pishbini[0]:

    i = i + 1
    if ehtemal > maximum:
        maximum = ehtemal

        index = i - 1

print(index)



# %% [markdown]
# MLP

# %%
train_images_flat = train_images.reshape(60000,28*28)
test_images_flat = test_images.reshape(10000, 28*28)
train_images

# %%
train_images_flat

# %%
from keras.layers import Input

# %%
MLP = Sequential()

MLP.add(Input(shape=(784,)))
MLP.add(Dense(units=256,activation='relu'))
MLP.add(Dense(units=128,activation='relu'))
MLP.add(Dense(units=10,activation='softmax'))

# %%
MLP.summary()

# %%
MLP.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# %%
hMLP = MLP.fit(train_images_flat,train_labels,epochs=10,validation_data=(test_images_flat,test_labels))

# %%
plt.plot(hCNN.history['accuracy'],label='CNN')
plt.plot(hMLP.history['accuracy'],label='MLP')
plt.legend()
plt.show()

# %%



