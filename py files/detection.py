import numpy as np
import extractor as et
import cv2 as cv
import keras


imagi = "sample1.png"
sels = et.cplit_b0rd_cells_np(imagi)
sel = sels[1]
et.image_displayer(sel)



def predict(img):
    image = img.copy()

    image = cv.resize(image, (28, 28))

    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255

    # plt.imshow(image.reshape(28, 28), cmap='Greys')
    # plt.show()
    et.image_displayer(image)
    model = keras.models.load_model('model/test_model.h5')
    pred = model.predict(image)
    return pred.argmax()


result = predict(sel)

