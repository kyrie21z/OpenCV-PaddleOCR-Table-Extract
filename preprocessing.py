import os.path

import cv2 as cv
# from ROI_selection import detect_lines, get_ROI
import numpy as np
from paddleocr import PPStructureV3, PaddleOCR


def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def get_binary(image):
    blackAndWhiteImage = cv.adaptiveThreshold(image, 255,
                                      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, 11, 2)
    return blackAndWhiteImage


def invert_area(image, x, y, w, h, display=False):
    ones = np.copy(image)
    ones = 1

    image[y:y + h, x:x + w] = ones * 255 - image[y:y + h, x:x + w]

    if (display):
        cv.imshow("inverted", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return image

def detect(cropped_frame, is_number=False):
    img = PPStructureV3()
    res = img.predict(cropped_frame)
    text = res[0]['overall_ocr_res']['rec_texts']
    return text


def draw_text(src, x, y, w, h, text):
    cFrame = np.copy(src)
    cv.rectangle(cFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.putText(cFrame, "text: " + text, (50, 50), cv.FONT_HERSHEY_SIMPLEX,
               2, (0, 0, 0), 5, cv.LINE_AA)

    return cFrame

def draw_rec(file_name, src):
    img = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
    )

    result = img.predict(src)

    res_ocr_file = file_name + '_ocr.jpg'
    res_json_file = file_name + '_ocr.json'
    res_ocr_path = os.path.join('./output', res_ocr_file)
    res_json_path = os.path.join('./output', res_json_file)
    for res in result:
        res.save_to_img(res_ocr_path)
        res.save_to_json(res_json_path)

    rec_list = []
    for coor in result[0]['dt_polys']:
        rec_list.append(coor)

    color = (255, 0, 0)

    for rec in rec_list:
        pt1 = (int(rec[0][0]), int(rec[0][1]))
        pt2 = (int(rec[2][0]), int(rec[2][1]))

        cv.rectangle(src, pt1, pt2, color)

    # return src


def erode(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_erosion = cv.dilate(img, kernel, iterations=2)
    return img_erosion

def preprocess(image):
    # 高斯模糊减少噪点
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    # 转换为灰度图
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # 自适应阈值处理
    thresh = cv.adaptiveThreshold(gray, 255,
                                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV, 11, 2)
    return thresh

def main():
    img_path = './photo/general_ocr_002.png'
    text = detect(img_path)
    print(text)

if __name__ == '__main__':
    main()