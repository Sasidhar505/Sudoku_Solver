import cv2 as cv
import numpy as np 
import extractor as et
import tensorflow 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D , MaxPool2D
from keras import backend as k

(X_train , y_train) , (X_test , y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0] , 28 , 28 , 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0] , 28 , 28 , 1).astype('float32')

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
num_classes = y_test.shape[0]



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range [0, 1]
X_train = (X_train / 255.0)
X_test = (X_test / 255.0)


model = Sequential()
model.add(Conv2D(32 , (3,3) , activation='relu' , kernel_initializer= 'he_uniform' , input_shape = (28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save("model/madel_ocr.h5")

print("Saved Model to Folder")


def predict(img):
    image = img.copy()

    image = cv.resize(image, (28, 28))

    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    model = keras.models.load_model('model/madel_ocr.h5')
    pred = model.predict(image, batch_size=1)
    return pred.argmax()


def img_invartar(iimage):
    iimage = (255-iimage)
    

imagi = "sample.png"
doku_sels = et.cplit_b0rd_cells_np(imagi)

sel = doku_sels[6]
gray = cv.threshold(sel, 128, 255, cv.THRESH_BINARY)[1]

# Find contours
cnts = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    kunt_img = cv.drawContours(gray.copy() , cnts , -1 , (0,255,0) , 3)
    cv.imshow('hi' , kunt_img)
    cv.waitKey(0)
    cv.destroyAllWindows
    if (x < 3 or y < 3 or h < 3 or w < 3):
        # Note the number is always placed in the center
        # Since image is 28x28
        # the number will be in the center thus x >3 and y>3
        # Additionally any of the external lines of the sudoku will not be thicker than 3
        
        ROI = gray[y:y + h, x:x + w]
    # increasing the size of the number allws for better interpreation,
    # try adjusting the number and you will see the differnce
        ROI = cv.resize(ROI, (100,100))
Rois = 255 - ROI
cv.imshow('hi' , Rois)
cv.waitKey(100)

result = predict(Rois)
print(result)




