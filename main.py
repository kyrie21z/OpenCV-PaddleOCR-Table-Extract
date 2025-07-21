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

    horizontal, vertical = detect_lines(rgb_src)


    keywords = ['对侧端子', '端子号', '功能描述']

    dict_table = {}
    for keyword in keywords:
        dict_table[keyword] = []

    ocr = PaddleOCR(
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_doc_orientation_classify=False
    )

    result = ocr.predict(img_path)

    idx = 0
    for word in result[0]['rec_texts']:
        if word == '端子号':
            break
        idx += 1
    key_coor = result[0]['rec_polys'][idx]
    min_y = min(l[1] for l in key_coor)
    max_y = max(l[1] for l in key_coor)

    cnt = 0
    for coor in result[0]['rec_polys']:
        temp_min_y = min(l[1] for l in coor)
        temp_max_y = max(l[1] for l in coor)
        if min_y < temp_min_y and max_y > temp_max_y:
            dict_table['端子号'].append(result[0]['rec_texts'][cnt])
        cnt += 1

    print(dict_table)

    return 0

if __name__ == '__main__':
    main()