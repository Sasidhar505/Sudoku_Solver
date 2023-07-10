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


def image_displayer(image):
    cv.imshow("see",image)
    cv.waitKey(0)
    cv.destroyAllWindows
    return




'''  -------------------------------------------------------  '''


def basic_processor(img_name , path = "Input_Data/") :
    img_path = path + str(img_name)
    base_img = cv.imread(img_path)
    image_displayer(base_img)
    gray_img = cv.cvtColor(base_img,cv.COLOR_BGR2GRAY)
    gray_img = cv.GaussianBlur(gray_img,(9,9),0)
    tresh_img = cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,17,3)
    bfilter = cv.bilateralFilter(tresh_img, 13, 20, 20)
    edged = cv.Canny(bfilter, 30, 180)
    return(edged , base_img)



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
    

    kunts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    kantors = imutils.grab_contours(kunts)
    kunt_img = cv.drawContours(base_img.copy() , kantors , -1 , (0,255,0) , 3) 
    image_displayer(kunt_img)
    kontors = sorted(kantors,key=cv.contourArea , reverse=True) [:17 ]
    lokshn = None

    for kontor in kontors:
        almost = cv.approxPolyDP(kontor , 15 , True)
        if len(almost) == 4 :
            lokshn = almost
        break
    preped_img = perspective_formar(base_img , lokshn )
    preped_img = cv.cvtColor(preped_img , cv.COLOR_BGR2GRAY)
    return preped_img , lokshn
    




'''

    the following function should be able to take image name as 
    argument and generate 81 zy.png files for all the corresponding 
    elements of your sudoku puzzle '''

def cplit_sdoku_cells(img_name):
    board , lokshn = sudoku_fienda(img_name)
    #cv.imwrite("Board" , board)
    image_displayer(board)
    length , girth = board.shape[:2]
    print(board.shape)
    row_start , col_start = int(0) , int(0)
    row_end , col_end = int(length/9) , int(girth/9)
    for ih in range(9):
        for iw in range(9):

            x = int(girth/9*iw )
            y = int(length/9*ih)
            h = int((length / 9))
            w = int((girth / 9 ))
            print(x,y,h,w)
            img = board[y:y+h, x:x+w]
            print(img.shape)
            #cv.imshow('hi',img)
            #img = (255-img)
            #cv.waitKey(50)
            cv.imwrite("BufferBox/" + str(ih) + str(iw) + ".png" , img)
            cv.destroyAllWindows
    return  
         

#cplit_sdoku_cells("sample1.png")

'''
    the following function should be able to take in numpy array 
    corresponding to the name of image and develeop individual
    numpy arrays of each cell present in the sudoku board'''

doku_board , lokshn = sudoku_fienda("sample1.png")
print(doku_board.shape)
print(doku_board[0].shape)
print(doku_board[0].reshape(27,27))
print(doku_board[0].shape)
imagi = "sample1.png"
def cplit_b0rd_cells_np(imag_nem):
    doku_board , lokshn = sudoku_fienda(imagi)
    sels = []
    roes = np.vsplit(doku_board,9)
    for z in roes:
        cals = np.hsplit(z,9)
        for sel in cals:
            sel = sel[7:70 , 7:70]
            sel = cv.resize(sel , (27,27) )/255.0
            cv.imshow('sels' , sel)
            cv.waitKey(50)
            sels.append(sel)

    return sels

cplit_b0rd_cells_np(imagi)




def img_invartar(iimage , name):
    iimage = (255-iimage)
    cv.imwrite(name , iimage)














