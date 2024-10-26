import os
import numpy as np
import cv2
from pytesseract import image_to_string
root = os.getcwd()


def getNumber(image):
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gry, 127, 255, cv2.THRESH_BINARY_INV)[1]
    return image_to_string(
        thr, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')


print("-- Check the predicted numbers with the numbers in the sudoku grid \n Press Y or y if you want to change the number of just press Enter\n")
def digit_recognizer(interactive_mode=False):
    sudoku_board = []
    for i,img_name in enumerate(sorted(os.listdir(os.path.join(root, 'temp')))):
        if '.png' not in img_name:
            continue
        img_path = os.path.join(root, 'temp', img_name)
#        print(img_path.split('/')[-1])
        row,col = i//9 + 1, i%9 + 1
        print(str(row) + ' x ' + str(col), sep = ' : ')

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        number = getNumber(cv2.imread(img_path)).replace('\x0c', '').strip()
        if np.all(img==0) or number not in list(map(str, range(0,10))):
            result = "."
        else:
            result = int(number)
        print(result)

        if interactive_mode:
            flag = input('Do you want to change the number ? (Y/N) :')
            if flag == 'y' or flag == 'Y':
                result = int(input("Enter the number :"))

        sudoku_board.append(result)

    return sudoku_board
