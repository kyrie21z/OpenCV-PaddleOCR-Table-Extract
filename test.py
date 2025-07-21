# from paddleocr import PaddleOCR, PPStructureV3
# import cv2 as cv
# import numpy as np
#
# ocr = PPStructureV3(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=True,
# )
#
# img_path = 'photo/general_ocr_002.png'
# result = ocr.predict(img_path)
# image = cv.imread(img_path)
#
# for res in result:
#     print(res)
#     res.save_to_img('./output')
#     res.save_to_json('./output')
#
# # angle = result[0]['doc_preprocessor_res']['angle']
#
# # rec_list = []
# # for coor in result[0]['overall_ocr_res']['dt_polys']:
# #     rec_list.append(coor)
# #
# # color = (0, 0, 255)
# # thickness = 1
# #
# # for i in range(len(rec_list)):
# #     pt1 = (int(rec_list[i][0][0]),int(rec_list[i][0][1]))
# #     pt2 = (int(rec_list[i][2][0]), int(rec_list[i][2][1]))
# #
# #     cv.rectangle(image, pt1, pt2, color, thickness)
# #
# # w, h = image.shape[:2]
# # resized_image = cv.resize(image, (h//2, w//2))
# #
# # cv.imwrite('./photo/processed_image.jpg', image)
# # result[0].save_to_img('./output')
# #
# # cv.imshow("words_rectangle", image)
# # cv.imshow("resized_image", resized_image)
# # cv.waitKey(0)
# # cv.destoryAllWindows()

import cv2
from paddleocr import PPStructureV3

# 初始化PaddleOCR（首次运行会自动下载模型）
ocr = PPStructureV3()  # 中文识别


# 使用OpenCV读取并处理图像
def process_image(image_path):
    img = cv2.imread(image_path)

    # 立即转换为RGB格式
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 在RGB格式下进行处理（示例：灰度化）
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 阈值二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 转换回RGB（保持三通道，便于PaddleOCR处理）
    final_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    return final_img


# 处理图像
processed_img = process_image("photo/general_ocr_002.png")

# 将OpenCV处理后的图像传递给PaddleOCR
result = ocr.predict(processed_img)

# 输出识别结果
print(result[0]['overall_ocr_res']['rec_texts'])