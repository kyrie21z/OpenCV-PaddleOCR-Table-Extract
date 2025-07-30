import os
import json

from paddleocr import PaddleOCR
import cv2 as cv
import numpy as np
from functions import detect_lines, perform_template_matching, remove_close_coordinates, get_triangle_context, detect_bottom_line, determine_cover_region
from preprocessing import preprocess
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

    # 处理图像并画线画框
    # 复制image
    image_copy = image
    # 获取水平线与竖直线
    horizontal, vertical = detect_lines(file_name, image_copy, result, is_draw=False)
    # 获取表格底线
    '''bottom_line = detect_bottom_line(vertical, result)

    bottom_line_x = bottom_line[0]
    print(vertical)
    print(bottom_line)'''

    idx = 0
    idx_list = []
    bottom_lines = []
    circle_template_matches_res, circle_accepted_centers_res = [], []
    roi_circle_list = []
    dzh_column_region= []
    for word in result[0]['rec_texts']:
        if word == '端子号':
            idx_list.append(idx)
        idx += 1
    for idx in idx_list:
        key_coor = result[0]['rec_polys'][idx]
        min_y = int(min(l[1] for l in key_coor))
        max_y = int(max(l[1] for l in key_coor))
        max_x = int(max(l[0] for l in key_coor))
        dist = max_y - min_y

        # 检测区域为端子号文字框右侧
        roi_circle = image[min_y: min_y + int(dist / 2), max_x:image.shape[1]]
        # 端子号检测区域
        roi_dzh = image[min_y - 20: max_y + 20, max_x: image.shape[1]]
        # cv.imshow('roi_circle', roi_circle)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        height, width = roi_dzh.shape[0:2]
        # 旋转roi以提升ocr识别率
        rotated_image = np.rot90(roi_dzh, k = -1)
        # cv.imshow('rotated_image', rotated_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        ocr_1 = PaddleOCR(
            lang='ch',
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_doc_orientation_classify=False
        )
        result_1 = ocr_1.predict(rotated_image)
        for res in result_1:
            res.save_to_img('photo')
            res.save_to_json('photo')
        last_elem_dzh = []
        idx_1 = 0
        for word in result_1[0]['rec_texts']:
            if not word.isdigit():
                last_elem_dzh = result_1[0]['rec_polys'][idx_1 - 1]
                break
            idx_1 += 1
        if len(last_elem_dzh) == 0:
            last_elem_dzh = result_1[0]['rec_polys'][idx_1 - 1]
        last_elem_transformed = []
        for coor in last_elem_dzh:
            point = [max_x + coor[1], min_y + height - coor[0]]
            last_elem_transformed.append(point)
        print("last_elem_transformed:")
        print(last_elem_transformed)
        bottom_line = detect_bottom_line(vertical, last_elem_transformed, result)
        bottom_line_x = bottom_line[0]
        # 将roi区域四个顶点坐标加入列表用于后续填色覆盖下小圆圈
        roi_circle_list.append(np.array([[max_x, min_y], [max_x, min_y + int(dist / 2)], [bottom_line_x, min_y + int(dist / 2)], [bottom_line_x, min_y]]))

        dzh_column_region.append(np.array([[max_x, min_y], [max_x, max_y], [bottom_line_x, max_y], [bottom_line_x, min_y]]))
        # cv.imshow('roi', roi)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # 根据模板匹配小圆圈
        for template in os.listdir('template_circle'):
            template_path = os.path.join('template_circle', template)
            template_brg = cv.imread(template_path)
            template_rgb = preprocess(cv.cvtColor(template_brg, cv.COLOR_BGR2RGB))

            roi_rgb = preprocess(roi_circle)


            template_matches, accepted_centers = perform_template_matching(roi_rgb, template_rgb)
            for rec in template_matches:
                circle_template_matches_res.append(rec)

            h, w = template_rgb.shape[0:2]
            for pt in accepted_centers:
                point = (int(pt[0]) + max_x, int(pt[1]) + min_y)
                circle_accepted_centers_res.append(point)
                cv.circle(image, point, int(h / 2), (0, 0, 255), 2)

    circle_accepted_centers_res = remove_close_coordinates(circle_accepted_centers_res, 6)
    cover_region = determine_cover_region(circle_accepted_centers_res,)
    # 填色覆盖小圆圈
    cv.fillPoly(image, roi_circle_list, [255, 255, 255])
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


    print(file_name)
    # print(template_matches)
    # print(accepted_centers)
    print(len(circle_accepted_centers_res))

    # cv.imwrite(f"output/{file_name}_res.png", image)

    # 根据模板匹配三角形
    triangle_template_matches_res, triangle_accepted_centers_res = [], []
    for template in os.listdir('template_triangle'):
        template_path = os.path.join('template_triangle', template)
        template_brg = cv.imread(template_path)
        template_rgb = preprocess(cv.cvtColor(template_brg, cv.COLOR_BGR2RGB))

        roi_rgb = preprocess(image)

        template_matches, accepted_centers = perform_template_matching(roi_rgb, template_rgb, 0.8)
        for rec in template_matches:
            triangle_template_matches_res.append(rec)

        h, w = template_rgb.shape[:2]
        for pt in accepted_centers:
            point = (int(pt[0]), int(pt[1]))
            triangle_accepted_centers_res.append(point)

    triangle_accepted_centers_res = remove_close_coordinates(triangle_accepted_centers_res, 10)
    triangle_context = {}
    for idx, pt in enumerate(triangle_accepted_centers_res):
        point = (int(pt[0]), int(pt[1]))
        # cv.circle(image, point, 2, (0, 0, 255), 2)
        triangle_context[idx] = get_triangle_context(point, result)

    print(len(triangle_accepted_centers_res))
    print(triangle_context)

    cv.imwrite(f"output/{file_name}_res.png", image)

    # 格式化数据
    keywords = ['对侧端子', '端子号', '功能描述']

    dict_table = {}

    #处理端子号
    idx = 0
    idx_list = []
    for idx, word in enumerate(result[0]['rec_texts']):
        if word == '端子号':
            idx_list.append(idx)
        idx += 1
    for i, idx in enumerate(idx_list):
        key_coor = result[0]['rec_polys'][idx]
        min_y = min(l[1] for l in key_coor)
        max_y = max(l[1] for l in key_coor)

        cnt = 0
        for coor in result[0]['rec_polys']:
            temp_min_y = min(l[1] for l in coor)
            temp_max_y = max(l[1] for l in coor)
            if min_y < temp_min_y and max_y > temp_max_y:
                dict_table[f'端子号{i}'].append(result[0]['rec_texts'][cnt])
            cnt += 1

    #处理对侧端子
    idx = 0
    idx_list = []
    for word in result[0]['rec_texts']:
        if word == '对侧端子':
            idx_list.append(idx)
        idx += 1
    for i, idx in enumerate(idx_list):
        key_coor = result[0]['rec_polys'][idx]
        min_y = min(l[1] for l in key_coor)
        max_y = max(l[1] for l in key_coor)

        cnt = 0
        for coor in result[0]['rec_polys']:
            temp_min_y = min(l[1] for l in coor)
            temp_max_y = max(l[1] for l in coor)
            if min_y-20 < temp_min_y and max_y+20 > temp_max_y and not temp_max_y == max_y and not temp_min_y == min_y:
                dict_table[f'对侧端子{i}'].append(result[0]['rec_texts'][cnt])
            cnt += 1

    #处理功能描述
    idx = 0
    idx_list = []
    for word in result[0]['rec_texts']:
        if word == '功能描述':
            idx_list.append(idx)
        idx += 1
    for i, idx in enumerate(idx_list):
        key_coor = result[0]['rec_polys'][idx]
        min_y = min(l[1] for l in key_coor)
        max_y = max(l[1] for l in key_coor)

        cnt = 0
        for coor in result[0]['rec_polys']:
            temp_min_y = min(l[1] for l in coor)
            temp_max_y = max(l[1] for l in coor)
            if min_y-100 < temp_min_y and max_y+100 > temp_max_y and not temp_max_y == max_y and not temp_min_y == min_y:
                dict_table[f'功能描述{i}'].append(result[0]['rec_texts'][cnt])
            cnt += 1

    res_file = file_name + '_res.json'
    res_path = os.path.join('./output', res_file)
    with open(res_path, 'w', encoding="utf-8") as json_file:
        json.dump(dict_table, json_file, indent=4, ensure_ascii=False)

    print(dict_table)

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
        img_cropped = img_rgb[50: img_rgb.shape[0] - 150, 200: img_rgb.shape[1] - 50]
        # img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
        # img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        file_name = f"page_{i}"
        main(img_cropped, file_name)
        if i == 0:
            break

    # for file in os.listdir('./photo'):
    #     path = os.path.join('./photo', file)
    #     filename, _ = os.path.splitext(file)
    #     main(filename, path)