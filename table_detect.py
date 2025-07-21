import cv2
import numpy as np


def detect_table(image_path):
    # 读取图像并进行预处理
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 边缘检测
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # 轮廓检测
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选表格轮廓
    table_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            table_contour = contour

    # 绘制表格轮廓
    if table_contour is not None:
        cv2.drawContours(image, [table_contour], -1, (0, 255, 0), 2)

        # 获取表格边界框
        x, y, w, h = cv2.boundingRect(table_contour)

        # 分析表格结构（行和列）
        # 这里只是一个示例，实际应用中可能需要更复杂的算法
        # 例如使用霍夫变换检测直线，或基于形态学操作分析表格结构

        return image, (x, y, w, h)
    else:
        return image, None


# 使用示例
if __name__ == "__main__":
    image_path = "photo/page_3.png"  # 替换为你的图片路径
    result_image, table_bbox = detect_table(image_path)

    if table_bbox is not None:
        print(f"检测到表格，位置：{table_bbox}")
    else:
        print("未检测到表格")

    # 显示结果
    cv2.imshow("Table Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()