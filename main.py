from paddleocr import PaddleOCR
import cv2 as cv
import numpy as np
from ROI_selection import detect_lines, get_ROI
from preprocessing import get_grayscale, get_binary, draw_text, erode, detect, draw_rec

def main(display = False, print_text = False, write = False):
    img_path = 'photo/page_7.png'

    src = cv.imread(img_path)
    rgb_src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

    gray = get_grayscale(rgb_src)
    bw = get_binary(gray)
    # rgb_bw = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)

    horizontal, vertical = detect_lines(rgb_src, minLinLength=300, display=True, write = False)

    '''
    keywords = ['对侧端子', '端子号', '功能描述']

    dict_table = {}
    for keyword in keywords:
        dict_table[keyword] = []

    #set boundaries
    first_line_index = 1
    last_line_index = 4

    cnt = 0

    for i in range(first_line_index, last_line_index):
        for j, keyword in enumerate(keywords):
           cnt += 1

           left_line_index = j
           right_line_index = j+1
           top_line_index = i
           bottom_line_index = i+1

           cropped_img, (x, y, w, h) = get_ROI(rgb_src, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index)

           # print(cropped_img)
           # text = detect(cropped_img)
           img = PaddleOCR()
           res = img.predict(cropped_img)
           text = res[0]['rec_texts']
           for word in text:
               dict_table[keyword].append(word)

    print(dict_table)
    '''
    return 0

if __name__ == '__main__':
    main()