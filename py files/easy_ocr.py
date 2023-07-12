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

model.compile( optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy'])

model.save('model/my_ocr_madel.keras')

print("Saved Model to Folder")


def predict(img):
    image = img.copy()

    image = cv.resize(image, (28, 28))

    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    model = keras.models.load_model('model/my_ocr_madel.keras')
    pred = model.predict(image.reshape(1,28,28,1), batch_size=1)
    return pred.argmax()


def img_invartar(iimage):
    iimage = (255-iimage)
    

imagi = "sample1.png"
doku_sels = et.cplit_b0rd_cells_np(imagi)

sel = doku_sels[2]
thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
# threshold the image
gray = cv.threshold(sel, thresh, 255, cv.THRESH_BINARY)[1]

# Find contours
cnts = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x, y, w, h = cv.boundingRect(c)
    print(x,y,w,h)
    if (x < 3 or y < 3 or h < 3 or w < 3):
        # Note the number is always placed in the center
        # Since image is 28x28
        # the number will be in the center thus x >3 and y>3
        # Additionally any of the external lines of the sudoku will not be thicker than 3
        continue
    ROI = gray[y:y + h, x:x + w]
    et.image_displayer(ROI)
    # increasing the size of the number allws for better interpreation,
    # try adjusting the number and you will see the differnce
    ROI = cv.resize(ROI,None, fx=120,fy=120)
    #et.image_displayer(ROI)
  
    result = predict(ROI)
    print(result)





