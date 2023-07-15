import numpy as np



def xtrt_nom(fael_name , base_path = ""):
    txt_path = base_path + fael_name 
    file = open(txt_path , "r")
    nombrs = file.read()
    print('numbers:' , nombrs)
    return nombrs



dizits = xtrt_nom("digits.txt")

#queshun = np.array(dizits).reshape(9,9)

print('hi' , dizits[10])




