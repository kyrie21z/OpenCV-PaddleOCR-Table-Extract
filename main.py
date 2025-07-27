import os
import json

from paddleocr import PaddleOCR
import cv2 as cv
import numpy as np
from functions import detect_lines, get_ROI, perform_template_matching
from preprocessing import get_grayscale, get_binary, draw_text, erode, detect, draw_rec, preprocess
from pdf2image import convert_from_path

def main(image, file_name, display = False, print_text = False, write = False):
    # 进行ocr识别,并筛选出ROI
    ocr = PaddleOCR(
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_doc_orientation_classify=False
    )

    result = ocr.predict(image)

    # for res in result:
    #     res.save_to_img(f"ocr_res/{file_name}_ocr.jpg")

    idx = 0
    idx_list = []
    template_matches_res, accepted_centers_res = [], []
    for word in result[0]['rec_texts']:
        if word == '端子号':
            idx_list.append(idx)
        idx += 1
    for idx in idx_list:
        key_coor = result[0]['rec_polys'][idx]
        min_y = int(min(l[1] for l in key_coor)) - 15
        max_y = int(max(l[1] for l in key_coor))
        dist = max_y - min_y
        roi = image[min_y: min_y + int(dist / 2), 0:image.shape[1]]
        # cv.imshow('roi', roi)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        #获取模板匹配
        for template in os.listdir('template'):
            template_path = os.path.join('template', template)
            template_brg = cv.imread(template_path)
            template_rgb = preprocess(cv.cvtColor(template_brg, cv.COLOR_BGR2RGB))

            roi_rgb = preprocess(roi)


            template_matches, accepted_centers = perform_template_matching(roi_rgb, template_rgb)
            for rec in template_matches:
                template_matches_res.append(rec)

            h, w = template_rgb.shape
            for pt in accepted_centers:
                point = (int(pt[0]), int(pt[1]) + min_y)
                accepted_centers_res.append(point)
                cv.circle(image, point, int(h / 2), (0, 0, 255), 2)

    print(file_name)
    # print(template_matches)
    # print(accepted_centers)
    print(len(accepted_centers_res))



    cv.imwrite(f"output/{file_name}_res.png", image)


    '''# 处理图像并画线画框
    src = cv.imread(img_path)
    rgb_src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

    gray = get_grayscale(rgb_src)
    bw = get_binary(gray)
    # rgb_bw = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)

    horizontal, vertical = detect_lines(file_name, rgb_src)


    keywords = ['对侧端子', '端子号', '功能描述']

    dict_table = {}
    for keyword in keywords:
        dict_table[keyword] = []

    #处理端子号
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

    #处理对侧端子
    idx = 0
    for word in result[0]['rec_texts']:
        if word == '对侧端子':
            break
        idx += 1
    key_coor = result[0]['rec_polys'][idx]
    min_y = min(l[1] for l in key_coor)
    max_y = max(l[1] for l in key_coor)

    cnt = 0
    for coor in result[0]['rec_polys']:
        temp_min_y = min(l[1] for l in coor)
        temp_max_y = max(l[1] for l in coor)
        if min_y-20 < temp_min_y and max_y+20 > temp_max_y and not temp_max_y == max_y and not temp_min_y == min_y:
            dict_table['对侧端子'].append(result[0]['rec_texts'][cnt])
        cnt += 1

    #处理功能描述
    idx = 0
    for word in result[0]['rec_texts']:
        if word == '功能描述':
            break
        idx += 1
    key_coor = result[0]['rec_polys'][idx]
    min_y = min(l[1] for l in key_coor)
    max_y = max(l[1] for l in key_coor)

    cnt = 0
    for coor in result[0]['rec_polys']:
        temp_min_y = min(l[1] for l in coor)
        temp_max_y = max(l[1] for l in coor)
        if min_y-100 < temp_min_y and max_y+100 > temp_max_y and not temp_max_y == max_y and not temp_min_y == min_y:
            dict_table['功能描述'].append(result[0]['rec_texts'][cnt])
        cnt += 1

    res_file = file_name + '_res.json'
    res_path = os.path.join('./output', res_file)
    with open(res_path, 'w', encoding="utf-8") as json_file:
        json.dump(dict_table, json_file, indent=4, ensure_ascii=False)

    print(dict_table)'''

    return 0

if __name__ == '__main__':
    pdf_path = 'input/drawings.pdf'
    images = convert_from_path(pdf_path)
    for i, img in enumerate(images):
        # if not i == 12:
        #     continue
        #将PIL转换成numpy数组
        img_np = np.array(img)
        img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
        # img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
        # img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        file_name = f"page_{i}"
        main(img_rgb, file_name)
        # if i == 0:
        #     break

    # for file in os.listdir('./photo'):
    #     path = os.path.join('./photo', file)
    #     filename, _ = os.path.splitext(file)
    #     main(filename, path)