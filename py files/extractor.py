'''lets try and divide the image into sections by this commit'''

'''this py file basically should take the image that user sends over
    and process it in a way so that 
    1) all the noise in the image is gone (gray,tresh)
    2) extract 9x9 sudoku puzzle 
    3) extract 1x1 squares for further ocr detection'''

import cv2 as cv
import numpy as np
# from scipy import misc
import imageio
import imutils
import PIL
import tensorflow as tf
import keras_ocr as ko
import matplotlib.pyplot as plt


def image_displayer(image):
    cv.imshow("see",image)
    cv.waitKey(0)
    cv.destroyAllWindows
    return




'''  -------------------------------------------------------  '''


def basic_processor(img_name , path = "Input_Data/") :
    img_path = path + str(img_name)
    base_img = cv.imread(img_path)
    #image_displayer(base_img)
    gray_img = cv.cvtColor(base_img,cv.COLOR_BGR2RGB)
    #image_displayer(gray_img)
    gray_img = cv.cvtColor(gray_img , cv.COLOR_RGB2GRAY)
    #gray_img = cv.GaussianBlur(gray_img,(9,9),0)
    tresh_img = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,39,10)
    #tresh_img = cv.bitwise_not(tresh_img , tresh_img)
    #kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    #tresh_img = cv.dilate(tresh_img, kernel)
    #image_displayer(tresh_img)
    
    #bfilter = cv.bilateralFilter(tresh_img, 13, 20, 20)
    #edged = cv.Canny(bfilter, 30, 180)
    #tresh_imgi = cv.bitwise_not(edged , edged)
    #image_displayer(tresh_img)
    return(tresh_img , base_img)



'''
    the following function should be able to take in our image ,
    sudoku location points and desired width and height and 
    return our board and preped_img '''

def perspective_formar(img_name , lokshn , height=729 , width = 729):
    putz1 = np.float32([lokshn[0], lokshn[3], lokshn[1], lokshn[2]])
    putz2 = np.float32([ [0,0] , [width,0] , [0,height] , [width,height]])

    metrix = cv.getPerspectiveTransform(putz1,putz2)
    preped_img = cv.warpPerspective(img_name , metrix , (width,height))
    return preped_img



''' 
    the following function should be able to detect the 9x9 sudoku
    and isolate the sudoku while warping it according to specified 
    dimensions using perspective_formar funcshun '''

def sudoku_fienda (img_name) :
    edged , base_img = basic_processor(img_name)
    print('in sudokufienda')
    #image_displayer(edged)

    kunts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    kantors = imutils.grab_contours(kunts)
    kunt_img = cv.drawContours(base_img.copy() , kantors , -1 , (0,255,0) , 3) 
    #image_displayer(kunt_img)
    kontors = sorted(kantors,key=cv.contourArea , reverse=True) [:17 ]
    lokshn = None

    for kontor in kontors:
        almost = cv.approxPolyDP(kontor , 15 , True)
        if len(almost) == 4 :
            lokshn = almost
        break
    preped_img = perspective_formar(base_img , lokshn )
    #preped_img = cv.cvtColor(preped_img , cv.COLOR_BGR2GRAY)
    #preped_img = cv.bitwise_not(preped_img , preped_img)
    print('board')
    #image_displayer(preped_img)
    return preped_img , lokshn
    





def cplit_b0rd_cells_np(imag_nem):
    doku_board , lokshn = sudoku_fienda(imag_nem)
    gray_img = cv.cvtColor(doku_board,cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(gray_img , cv.COLOR_RGB2GRAY)
    #image_displayer(gray_img)
    doku_b0rd = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,29,10)
    print('board in cplit')
    #image_displayer(doku_b0rd)
    sels = []
    roes = np.vsplit(doku_b0rd,9)
    for z in roes:
        cals = np.hsplit(z,9)
        for sel in cals:
            #sel = sel[7:70 , 7:70]
            #sel = (sel)/255.0
            #cv.imshow('sels' , sel)
            #cv.waitKey(50)
            sels.append(sel)

    return sels


# sels = cplit_b0rd_cells_np("sample1.png")
# print(sels[1].shape)
# image_displayer(sels[1])





'''
    Predicshun : following function tries to predict the 
    digits from the numpy array directly using 
    keras model model_ocr.h5 instead of using easy_ocr 
    which took in images previously'''

# def predicshun_of_npsels(nparr_nem):
#     pipline = ko.pipeline.Pipeline()
#     boku_sels = cplit_b0rd_cells_np(imagi)
#     doku_zels = np.array(boku_sels).reshape(-1 , 9 , 9 , 1)
#     classes = np.arange(0,10)
#     model = tf.keras.models.load_model("model/digits.h5")

#     predctd_nems = []
#     for k in predctd_nems:
#         index = (np.argmax(k))
#         predctd_nem = classes[index]
#         predctd_nems.append(predctd_nem)
#     print(predctd_nems)

#     return


# predicshun_of_npsels(imagi)




















