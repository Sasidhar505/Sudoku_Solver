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

def trn_sev_modl():
    (X_train , y_train) , (X_test , y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0] , 28 , 28 , 1)
    X_test = X_test.reshape(X_test.shape[0] , 28 , 28 , 1)

    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train,num_classes)
    y_test = keras.utils.to_categorical(y_test,num_classes)
   


    model = Sequential()
    model.add(Conv2D(32 , (3,3) , activation='relu'  , input_shape = (28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile( loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])

    batch_size = 128
    epochs = 10

    model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("model/test_model.h5")
    return score[0] , score[1]


#loss , accuracy = trn_sev_modl()

def predict(img):
    image = img.copy()

    image = cv.resize(image, (28, 28))

    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    model = keras.models.load_model('model/digits.h5')
    pred = model.predict(image.reshape(1,28,28,1), batch_size=1)
    return pred.argmax()


def img_invartar(iimage):
    iimage = (255-iimage)
    

imagi = "sample.png"
doku_sels = et.cplit_b0rd_cells_np(imagi)

sel = doku_sels[13]
sel = sel[3:78 , 3:78]
et.image_displayer(sel)
thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
# threshold the image

#gray = cv.bitwise_not(gray , gray)
# Find contours
hi = predict(sel)
print(hi)
gray = cv.threshold(sel, thresh, 255, cv.THRESH_BINARY)[1]
cnts = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
    ROI_r = cv.resize(ROI,None, fx=5,fy=5)
    
    et.image_displayer(ROI_r)
    
    result = predict(ROI_r)
    print(result)
    break




