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
    dizit = reader.readtext(str(image_path) , detail = 0)
    #print(dizit)
    return dizit




def list_dizits(image_name):
    et.cplit_sdoku_cells(image_name)
    dizit_lizt = []
    for z in range(9):
        for y in range(9):
            strinp = str(z)+str(y)
            dizit = prediction(strinp)
            if dizit is None:
                dizit_lizt.append(0)
            else :
                dizit_lizt.append(dizit)
    print(len(dizit_lizt))
    return dizit_lizt


list_dizits("sample.png")

