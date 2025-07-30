import os
import sys
import math
import cv2 as cv
import numpy as np
from langchain_community.callbacks.tracers.wandb import build_tree
from more_itertools.more import first
from paddle.base.libpaddle.eager.ops.legacy import pixel_shuffle
from paddleocr import PaddleOCR
from scipy.spatial import cKDTree

from preprocessing import draw_rec, preprocess

def is_vertical(line, threshold=3):
    return abs(line[0] - line[2]) <= threshold


def is_horizontal(line, threshold=3):
    return abs(line[1] - line[3]) <= threshold


def overlapping_filter(lines, sorting_index, threshold=5):
    filtered_lines = []

    # sorting_index: vertical -> 0 -> x, horizontal -> 1 -> y
    lines = sorted(lines, key=lambda lines: lines[sorting_index])

    cnt = 0
    for i in range(len(lines)):
        l_curr = lines[i]
        if (i > 0):
            l_prev = filtered_lines[cnt]
            if (abs(l_curr[sorting_index] - l_prev[sorting_index]) > threshold):
                filtered_lines.append(l_curr)
                cnt += 1
            else:
                first_point_1 = filtered_lines[cnt][sorting_index]
                first_point_2 = l_curr[sorting_index]
                last_point_1 = filtered_lines[cnt][sorting_index+2]
                last_point_2 = l_curr[sorting_index+2]
                filtered_lines[cnt][sorting_index] = min(first_point_1, first_point_2)
                filtered_lines[cnt][sorting_index+2] = max(last_point_1, last_point_2)
        else:
            filtered_lines.append(l_curr)

    return filtered_lines


# def detect_lines(image, title='default', rho=1, theta=np.pi / 180, threshold=50, minLinLength=500, maxLineGap=6,
#                  display=False, write=False):
#     # Check if image is loaded fine
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#     if gray is None:
#         print('Error opening image!')
#         return -1
#
#     dst = cv.Canny(gray, 50, 150, None, 3)
#
#     # Copy edges to the images that will display the results in BGR
#     cImage = np.copy(image)
#
#     # linesP = cv.HoughLinesP(dst, 1 , np.pi / 180, 50, None, 290, 6)
#     linesP = cv.HoughLinesP(dst, rho, theta, threshold, None, minLinLength, maxLineGap)
#
#     horizontal_lines = []
#     vertical_lines = []
#
#     if linesP is not None:
#         # for i in range(40, nb_lines):
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#
#             if (is_vertical(l)):
#                 vertical_lines.append(l)
#
#             elif (is_horizontal(l)):
#                 horizontal_lines.append(l)
#
#         horizontal_lines = overlapping_filter(horizontal_lines, 1)
#         vertical_lines = overlapping_filter(vertical_lines, 0)
#
#     if (display):
#         for i, line in enumerate(horizontal_lines):
#             cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3, cv.LINE_AA)
#
#             cv.putText(cImage, str(i) + "h", (line[0] + 5, line[1]), cv.FONT_HERSHEY_SIMPLEX,
#                        0.5, (0, 0, 0), 1, cv.LINE_AA)
#
#         for i, line in enumerate(vertical_lines):
#             cv.line(cImage, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv.LINE_AA)
#             cv.putText(cImage, str(i) + "v", (line[0], line[1] + 5), cv.FONT_HERSHEY_SIMPLEX,
#                        0.5, (0, 0, 0), 1, cv.LINE_AA)
#
#         draw_rec(cImage)
#         cv.imwrite('./photo/detected_image.jpg', cImage)
#         cv.imshow("Source", cImage)
#         # cv.imshow("Canny", cdstP)
#         cv.waitKey(0)
#         cv.destroyAllWindows()
#
#     if (write):
#         cv.imwrite("../Images/" + title + ".png", cImage);
#
#     return (horizontal_lines, vertical_lines)

def detect_lines(file, image, result, min_length=50, y_threshold=5, gapthreshold=10, continuity_threshold=0.95, is_draw = True):
    rec_list = result[0]['dt_polys']

    image_preprocessed = preprocess(image)

    # gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # binary = cv.adaptiveThreshold(gray, 255,
    #                                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv.THRESH_BINARY_INV, 11, 2)

    kernel_horizontal = cv.getStructuringElement(cv.MORPH_RECT, (min_length, 1))
    horizontal_lines_map = cv.morphologyEx(image_preprocessed, cv.MORPH_OPEN, kernel_horizontal)

    temp_lines = []
    processed_mask = np.zeros(horizontal_lines_map.shape, dtype=bool)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if horizontal_lines_map[y, x] > 0 and not processed_mask[y, x]:
                x_start, x_end = x, x

                while x_end + 1 < image.shape[1] and horizontal_lines_map[y, x_end + 1] > 0:
                    x_end += 1

                processed_mask[y, x_start:x_end + 1] = True

                segment_length = x_end - x_start + 1
                line_slice = image_preprocessed[y, x_start:x_end + 1]
                pixel_count = np.count_nonzero(line_slice)
                ratio = pixel_count / segment_length

                if ratio >= continuity_threshold:
                    temp_lines.append((x_start, y, x_end, y, 0))

    if not temp_lines:
        return

    temp_lines.sort(key=lambda l: [l[1]])
    merged_lines = []
    current_group = [temp_lines[0]]

    for i in range(1, len(temp_lines)):
        line = temp_lines[i]
        min_x1 = min(l[0] for l in current_group)
        max_x2 = max(l[2] for l in current_group)
        if abs(line[1] - current_group[-1][1]) <= y_threshold and (abs(line[0] - max_x2) < gapthreshold or abs(line[2] - min_x1) < gapthreshold):
            current_group.append(line)
        else:
            avg_y = int(np.mean([l[1] for l in current_group]))
            merged_lines.append((min_x1, avg_y, max_x2, avg_y))
            current_group = [line]

    if current_group:
        min_x1 = min(l[0] for l in current_group)
        max_x2 = max(l[2] for l in current_group)
        avg_y = int(np.mean([l[1] for l in current_group]))
        merged_lines.append((min_x1, avg_y, max_x2, avg_y))

    kernel_vertical = cv.getStructuringElement(cv.MORPH_RECT, (1, min_length))
    vertical_lines_map = cv.morphologyEx(image_preprocessed, cv.MORPH_OPEN, kernel_vertical)
    Temp_lines = []
    Processed_mask = np.zeros(vertical_lines_map.shape, dtype=bool)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            if vertical_lines_map[y, x] > 0 and not Processed_mask[y, x]:
                y_start, y_end = y, y
                while y_end + 1 < image.shape[0] and vertical_lines_map[y_end + 1, x] > 0:
                    y_end += 1

                Processed_mask[y_start:y_end + 1, x] = True

                segment_length = y_end - y_start + 1
                line_slice = image_preprocessed[y_start:y_end+1, x]
                pixel_count = np.count_nonzero(line_slice)
                ratio = pixel_count / segment_length

                if ratio >= continuity_threshold:
                    Temp_lines.append((x, y_start, x, y_end, 0))

    if not Temp_lines:
        return

    Temp_lines.sort(key=lambda l: [l[0]])
    Merged_lines = []
    Current_group = [Temp_lines[0]]

    for i in range(1, len(Temp_lines)):
        line = Temp_lines[i]
        min_y1 = min(l[1] for l in Current_group)
        max_y2 = max(l[3] for l in Current_group)
        if abs(line[0] - Current_group[-1][0]) <= y_threshold  and (abs(line[1] - max_y2) < gapthreshold or abs(line[3] - min_y1) < gapthreshold):
            Current_group.append(line)
        else:
            avg_x = int(np.mean([l[0] for l in Current_group]))
            Merged_lines.append((avg_x, min_y1, avg_x, max_y2))
            Current_group = [line]

    if Current_group:
        min_y1 = min(l[1] for l in Current_group)
        max_y2 = max(l[3] for l in Current_group)
        avg_x = int(np.mean([l[0] for l in Current_group]))
        Merged_lines.append((avg_x, min_y1, avg_x, max_y2))

    horizontal_lines = merged_lines
    vertical_lines = Merged_lines

    if is_draw:
        # 画线
        for i, line in enumerate(horizontal_lines):
            cv.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3, cv.LINE_AA)

            cv.putText(image, str(i) + "h", (line[0] + 5, line[1]), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv.LINE_AA)

        for i, line in enumerate(vertical_lines):
            cv.line(image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv.LINE_AA)
            cv.putText(image, str(i) + "v", (line[0], line[1] + 5), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv.LINE_AA)

        #画框
        color = (255, 0, 0)

        for rec in rec_list:
            pt1 = (int(rec[0][0]), int(rec[0][1]))
            pt2 = (int(rec[2][0]), int(rec[2][1]))

            cv.rectangle(image, pt1, pt2, color)

        # 将结果写入输出文件
        res_file = file + '_res.jpg'
        path = os.path.join('./output', res_file)
        cv.imwrite(path, image)
        # cv.imshow("Source", image)
        # cv.imshow("Canny", cdstP)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return horizontal_lines, vertical_lines

def detect_bottom_line(vertical_lines, last_elem_rec, result):
    bottom_line = []
    max_x = max(l[0] for l in last_elem_rec)
    max_y = max(l[1] for l in last_elem_rec)
    min_x = min(l[0] for l in last_elem_rec)
    min_y = min(l[1] for l in last_elem_rec)
    print((min_x, min_y))
    print((max_x, max_y))
    for line in vertical_lines:
        if line[0] - min_x < 50 and max_x < line[0] + 10 and max_y < line[3] and min_y > line[1]:
            bottom_line = line
            break
    print(bottom_line)
    return bottom_line #[x1, y1, x2, y2]


def get_cropped_image(image, x, y, w, h):
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

# def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index,
#             offset=4):
#     x1 = vertical[left_line_index][2] + offset
#     y1 = horizontal[top_line_index][3] + offset
#     x2 = vertical[right_line_index][2] - offset
#     y2 = horizontal[bottom_line_index][3] - offset
#
#     w = x2 - x1
#     h = y2 - y1
#
#     cropped_image = get_cropped_image(image, x1, y1, w, h)
#
#     return cropped_image, (x1, y1, w, h)

def perform_template_matching(img, template, threshold = 0.68, overlap_threshold_ratio = 0.5, scales = (0.88, 1.0, 1.05)):
    h, w = template.shape
    template_matches = []
    accepted_centers = []
    for scale in scales:
        resized_template = cv.resize(template, (int(w * scale), int(h * scale)))
        result = cv.matchTemplate(img, resized_template, cv.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        # print(loc)
        potential_matches = []
        for pt in zip(*loc[::-1]):
            potential_matches.append((pt[0], pt[1]))


        for pt in potential_matches:
            x1, y1 = pt
            x2, y2 = x1 + w, y1 + h
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            is_duplicate = False
            for acc_cx, acc_cy in accepted_centers:
                if abs(cx - acc_cx) < w * overlap_threshold_ratio and abs(cy - acc_cy) < h * overlap_threshold_ratio:
                    is_duplicate = True
                    break
            if not is_duplicate:
                template_matches.append((x1, y1, x2, y2))
                accepted_centers.append((cx, cy))

    return template_matches, accepted_centers


def remove_close_coordinates(coordinates, threshold):
    """
    高效移除彼此接近的坐标点

    参数:
    coordinates (list): 包含坐标元组的列表，例如[(x1, y1), (x2, y2), ...]
    threshold (float): 判断坐标是否接近的距离阈值

    返回:
    list: 过滤后的坐标列表
    """
    if not coordinates:
        return []

    # 把坐标转换为NumPy数组
    coords_array = np.array(coordinates)

    # 构建KD树
    tree = cKDTree(coords_array)

    # 查找每个点距离在阈值内的其他点
    pairs = tree.query_pairs(threshold)

    # 确定要移除的点
    to_remove = set()
    for i, j in pairs:
        if i not in to_remove:
            to_remove.add(j)  # 选择移除第二个点，这可以根据需求调整

    # 创建过滤后的坐标列表
    filtered_coordinates = [coord for idx, coord in enumerate(coordinates) if idx not in to_remove]

    return filtered_coordinates

# 获取三角形指向文本， 并返回列表
def get_triangle_context(center, result):
    min_x = center[0]
    max_x = min_x + 600
    min_y = center[1] - 60
    max_y = center[1] + 60
    context_list = []
    for idx, coor in enumerate(result[0]['rec_polys']):
        coor_min_x = min(p[0] for p in coor)
        coor_max_x = max(p[0] for p in coor)
        coor_min_y = min(p[1] for p in coor)
        coor_max_y = max(p[1] for p in coor)
        if coor_min_y > min_y and coor_max_y < max_y and coor_min_x > min_x and coor_max_x < max_x:
            context_list.append(result[0]['rec_texts'][idx])

    return context_list

def determine_cover_region()