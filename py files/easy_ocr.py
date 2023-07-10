import cv2 as cv
import numpy as np 
import extractor as et
import tensorflow 
import keras
import easyocr




#et.image_displayer(image)
def prediction(diz_num, path = "BufferBox/"):
    reader = easyocr.Reader(['ch_sim','en'])
    image_path = path + diz_num + ".png"
    img = cv.imread(image_path)
    crop_img = img[7:72 , 7:72]
    #et.image_displayer(crop_img)
    dizit = reader.readtext(str(image_path) , detail = 0)
    #print(dizit[0])
    #dizit_lizt = np.empty(1 , dtype=object, order='C', like=None)
    #dizit_lizt.insert(dizit)
    return dizit

#et.cplit_sdoku_cells("sample1.png")
#prediction("73")


def list_dizits(image_name):
    et.cplit_sdoku_cells(image_name)
    dizit_lizt = np.empty((9,9), dtype=int, order='C', like=None)
    for z in range(9):
        for y in range(9):
            strinp = str(z)+str(y)
            dizit = prediction(strinp)
            if dizit == []:
                dizit_lizt[z][y] = 0
            else :
                dizit_lizt[z][y] = dizit[0]
    print(len(dizit_lizt))
    print(dizit_lizt)
    return dizit_lizt


def txt_fiel_io(dizit_lizt , path = "digits.txt"):
    #nue_dizit_lizt = dizit_lizt.resize(9,9)
    fiel = open(path , "w+")
    queshun = str(dizit_lizt)
    fiel.write(queshun)
    fiel.close()


dizit_lizt = list_dizits("sample1.png")
#izzit_list_np = np.array(dizit_lizt)
txt_fiel_io(dizit_lizt)



