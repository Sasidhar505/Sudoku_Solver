
'''

    the following function should be able to take image name as 
    argument and generate 81 zy.png files for all the corresponding 
    elements of your sudoku puzzle '''

# def cplit_sdoku_cells(img_name):
#     board , lokshn = sudoku_fienda(img_name)
#     #cv.imwrite("Board" , board)
#     #image_displayer(board)
#     length , girth = board.shape[:2]
#     print(board.shape)
#     row_start , col_start = int(0) , int(0)
#     row_end , col_end = int(length/9) , int(girth/9)
#     for ih in range(9):
#         for iw in range(9):

#             x = int(girth/9*iw )
#             y = int(length/9*ih)
#             h = int((length / 9))
#             w = int((girth / 9 ))
#             print(x,y,h,w)
#             img = board[y:y+h, x:x+w]
#             #print(img.shape)
#             #cv.imshow('hi',img)
#             #img = (255-img)
#             #cv.waitKey(50)
#             cv.imwrite("BufferBox/" + str(ih) + str(iw) + ".png" , img)
#             cv.destroyAllWindows
#     return  
         

#cplit_sdoku_cells("sample1.png")

'''
    the following function should be able to take in numpy array 
    corresponding to the name of image and develeop individual
    numpy arrays of each cell present in the sudoku board'''

# doku_board , lokshn = sudoku_fienda("sample1.png")
# print(doku_board.shape)
# print(doku_board[0].shape)
# print(doku_board[0].reshape(27,27))
# print(doku_board[0].shape)





# cnts = cv.findContours(sel, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]


# for c in cnts:
#     x, y, w, h = cv.boundingRect(c)
#     #print(x,y,w,h)
#     if (x < 3 or y < 3 or h < 3 or w < 3):
#         # Note the number is always placed in the center
#         # Since image is 28x28
#         # the number will be in the center thus x >3 and y>3
#         # Additionally any of the external lines of the sudoku will not be thicker than 3
#         continue
#     ROI = sel[y-4:y + h +4, x-4:x + w+4]
#     et.image_displayer(ROI)
#     #ROI = cv.resize(ROI , (120,120))
#     #print(x,y,w,h)
#     # increasing the size of the number allws for better interpreation,
#     # try adjusting the number and you will see the differnce
#     resu = predict(ROI)
#     print('withcnt: ' , resu)
#     break
