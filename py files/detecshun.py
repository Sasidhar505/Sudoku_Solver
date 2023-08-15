import cv2 as cv
import numpy as np 
import extractor as et
import tensorflow 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D , MaxPool2D
from keras.optimizers import RMSprop
from keras import backend as k
import pytesseract
#import sudoku_solver as ss
def trn_sev_modl():
    (X_train , y_train) , (X_test , y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0] , 28 , 28 , 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0] , 28 , 28 , 1).astype('float32')



    
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    model.save("model/mnist.h5")
    return 


#loss = trn_sev_modl()

def predict(img):
    image = img.copy()
    image = image[7:73 , 7:73]
    #et.image_displayer(image)
    image = cv.resize(image, (28, 28))

    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    if cv.countNonZero(image)  == 0 :
        return 0
    else:
        model = keras.models.load_model('model\mnist.h5')
        pred = model.predict(image.reshape(1,28,28,1), batch_size=1)
        #print(pred.argmax())
        return pred.argmax()


def img_invartar(iimage):
    iimage = (255-iimage)

def getNumber(image):
    #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Otsu Tresholding automatically find best threshold value
    image = image[7:73 , 7:73]
    
    # invert the image if the text is white and background is black
    if cv.countNonZero(image)  == 0 :
        return 0
    else :
        _, binary_image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
        count_white = np.sum(binary_image > 0)
        count_black = np.sum(binary_image == 0)
        if count_black > count_white:
            binary_image = 255 - binary_image
            
        # padding
        final_image = cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(255, 255, 255))
        et.image_displayer(final_image)
        txt = pytesseract.image_to_string(
            final_image, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
        return txt

    
doku_sels = et.cplit_b0rd_cells_np("sample.png")
sel = doku_sels[9]
# et.image_displayer(sel)
# print(predict(sel))
queshn = [0]*81

def build_b0rd(img_name):
    doku_sels = et.cplit_b0rd_cells_np(img_name)
    for z in range(81):
        queshn[z] = int(getNumber(doku_sels[z]))
    

    integer_array = np.array(queshn, dtype=np.int32)
    #print(integer_array)
    queshnn = integer_array.reshape((9,9))
    #print(queshnn.shape)
    return queshnn
        


et.image_displayer(sel)
x = getNumber(sel)
print( x , type(x))


# ss.print_board(queshnn)
# answer = ss.solve(queshnn)
# print("___________________")
# ss.print_board(answer)