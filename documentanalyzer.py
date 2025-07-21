import cv2
import numpy as np
from paddleocr import PaddleOCR


class DocumentAnalyzer:
    def __init__(self, image, template=None, lang='ch'):
        self.image = image
        self.template = template
        self.template_matches = []
        self.template_texts = {}
        self.merged_lines = []
        self.lines = []
        self.line_indices = {}
        self.box_to_line_text_boxes = {}
        self.boundary_extensions = []
        self.keyword_min_x = 0
        self.keyword_max_x = 0
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def perform_template_matching(self, threshold=0.71, overlap_threshold_ratio=0.5):
        """执行模板匹配，识别图像中的特定模板区域"""
        if self.template is None:
            return

        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        h, w = template_gray.shape
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)

        potential_matches = []
        for pt in zip(*loc[::-1]):
            potential_matches.append((pt[0], pt[1]))

        self.template_matches = []
        accepted_centers = []

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
                self.template_matches.append((x1, y1, x2, y2))
                accepted_centers.append((cx, cy))

                # 使用PaddleOCR检测模板附近的文本
                roi = self.image[y1:y2, x1:x2]
                if roi.size > 0:
                    texts = self._ocr_text(roi)
                    self.template_texts[(x1, y1, x2, y2)] = texts

    def detect_lines(self, min_length=20, y_threshold=5, gap_threshold=10, continuity_threshold=0.95):
        """检测图像中的水平线段"""
        if not self.template_matches:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length, 1))
        horizontal_lines_map = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)

        temp_lines = []
        processed_mask = np.zeros(horizontal_lines_map.shape, dtype=bool)

        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                if horizontal_lines_map[y, x] > 0 and not processed_mask[y, x]:
                    x_start, x_end = x, x

                    while x_end + 1 < self.image.shape[1] and horizontal_lines_map[y, x_end + 1] > 0:
                        x_end += 1

                    processed_mask[y, x_start:x_end + 1] = True

                    segment_length = x_end - x_start + 1
                    line_slice = binary[y, x_start:x_end + 1]
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
            if abs(line[1] - current_group[-1][1]) <= y_threshold:
                current_group.append(line)
            else:
                min_x1 = min(l[0] for l in current_group)
                max_x2 = max(l[2] for l in current_group)
                avg_y = int(np.mean([l[1] for l in current_group]))
                merged_lines.append((min_x1, avg_y, max_x2, avg_y))
                current_group = [line]

        if current_group:
            min_x1 = min(l[0] for l in current_group)
            max_x2 = max(l[2] for l in current_group)
            avg_y = int(np.mean([l[1] for l in current_group]))
            merged_lines.append((min_x1, avg_y, max_x2, avg_y))

        self.merged_lines = merged_lines

    def filter_and_associate_lines(self, scan_length=6000):
        """基于模板匹配结果筛选线段并建立关联"""
        if not self.template_matches or not self.merged_lines:
            return

        lines_to_keep = []

        for line in self.merged_lines:
            lx1, ly, lx2, _ = line
            is_associated_with_template = False

            for endpoint_x in [lx1, lx2]:
                for template_box in self.template_matches:
                    tx1, ty1, tx2, ty2 = template_box

                    is_horizontally_aligned = (tx1 <= endpoint_x <= tx2)
                    is_template_below = (ty1 > ly)
                    vertical_distance = ty1 - ly

                    if is_horizontally_aligned and is_template_below and vertical_distance < scan_length:
                        is_associated_with_template = True
                        break

                if is_associated_with_template:
                    break

            if is_associated_with_template:
                lines_to_keep.append(line)

        self.merged_lines = lines_to_keep

    def extend_boundaries_and_associate_text(self):
        """扩展线段边界并关联文本"""
        if not self.merged_lines:
            return

        self.lines = []
        self.line_indices = {}
        line_index = 1
        processed_pixels = set()

        for x1, y1, x2, y2 in self.merged_lines:
            self.lines.append((x1, y1, x2, y2))
            self.line_indices[(x1, y1, x2, y2)] = line_index

            for endpoint_x, line_type in [(x1, 'right_boundary'), (x2, 'left_boundary')]:
                closest_box = self.find_closest_text_box(endpoint_x, y1)
                closest_text = self.get_text_from_box(closest_box)
                distance = self.find_distance(endpoint_x, y1, closest_box)

                self.box_to_line_text_boxes[distance] = {
                    'box': closest_box,
                    'text': closest_text,
                    'distance': distance,
                    'line_type': line_type,
                    'line_index': line_index,
                    'template_texts': []
                }

            # 垂直扩展
            self.extend_vertically(x1, y1, processed_pixels, line_index)
            self.extend_vertically(x2, y2, processed_pixels, line_index)

            # 右侧边界扩展
            if x2 > self.keyword_max_x + 10:
                path, endpoint = self.extend_boundary_line_downwards(x2, y2)
                if len(path) > 1:
                    template_box, texts = self.find_template_region_for_point(endpoint[0], endpoint[1])
                    self.boundary_extensions.append({
                        'path': path,
                        'endpoint': endpoint,
                        'line_index': line_index,
                        'template_box': template_box,
                        'template_texts': texts
                    })
                    if texts:
                        self.associate_texts_with_line(line_index, texts)

            line_index += 1

    def _ocr_text(self, image):
        """使用PaddleOCR识别图像中的文本"""
        result = self.ocr.ocr(image, cls=True)
        texts = []
        for line in result:
            for box_info in line:
                bbox = box_info[0]
                text = box_info[1][0]
                confidence = box_info[1][1]
                texts.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        return texts

    def detect_text_near_template(self, box):
        """检测模板区域附近的文本"""
        x1, y1, x2, y2 = box
        # 扩展ROI区域以包含模板附近的文本
        expand_pixels = 50
        roi_x1 = max(0, x1 - expand_pixels)
        roi_y1 = max(0, y1 - expand_pixels)
        roi_x2 = min(self.image.shape[1], x2 + expand_pixels)
        roi_y2 = min(self.image.shape[0], y2 + expand_pixels)

        roi = self.image[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return []

        texts = self._ocr_text(roi)
        # 调整文本框坐标为原图坐标
        for text in texts:
            for point in text['bbox']:
                point[0] += roi_x1
                point[1] += roi_y1

        return texts

    def find_closest_text_box(self, x, y):
        """查找距离给定点最近的文本框"""
        min_distance = float('inf')
        closest_box = None

        for template_box in self.template_matches:
            texts = self.template_texts.get(template_box, [])
            for text in texts:
                bbox = text['bbox']
                # 计算点到文本框中心的距离
                box_center_x = sum([p[0] for p in bbox]) / 4
                box_center_y = sum([p[1] for p in bbox]) / 4
                distance = np.sqrt((x - box_center_x) ** 2 + (y - box_center_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_box = bbox

        return closest_box

    def get_text_from_box(self, box):
        """从文本框中提取文本"""
        if box is None:
            return ""

        for template_box in self.template_matches:
            texts = self.template_texts.get(template_box, [])
            for text in texts:
                bbox = text['bbox']
                # 判断两个框是否重叠（简化判断）
                if (abs((sum([p[0] for p in bbox]) / 4) - (sum([p[0] for p in box]) / 4)) < 20 and
                        abs((sum([p[1] for p in bbox]) / 4) - (sum([p[1] for p in box]) / 4)) < 20):
                    return text['text']

        return ""

    def find_distance(self, x, y, box):
        """计算点到文本框的距离"""
        if box is None:
            return float('inf')

        box_x = [p[0] for p in box]
        box_y = [p[1] for p in box]

        box_left = min(box_x)
        box_right = max(box_x)
        box_top = min(box_y)
        box_bottom = max(box_y)

        # 判断点是否在框内
        if box_left <= x <= box_right and box_top <= y <= box_bottom:
            return 0

        # 计算点到框边缘的最小距离
        dx = min(abs(x - box_left), abs(x - box_right))
        dy = min(abs(y - box_top), abs(y - box_bottom))

        if box_left <= x <= box_right:
            return dy
        elif box_top <= y <= box_bottom:
            return dx
        else:
            return np.sqrt(dx ** 2 + dy ** 2)

    def extend_vertically(self, x, y, processed_pixels, line_index):
        """垂直扩展线段"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # 向上扩展
        path_up = []
        current_y = y
        while current_y > 0 and binary[current_y, x] > 0 and (x, current_y) not in processed_pixels:
            path_up.append((x, current_y))
            processed_pixels.add((x, current_y))
            current_y -= 1

        # 向下扩展
        path_down = []
        current_y = y
        while current_y < binary.shape[0] - 1 and binary[current_y, x] > 0 and (x, current_y) not in processed_pixels:
            path_down.append((x, current_y))
            processed_pixels.add((x, current_y))
            current_y += 1

        # 记录扩展路径
        if path_up or path_down:
            path = path_up[::-1] + [(x, y)] + path_down
            self.boundary_extensions.append({
                'path': path,
                'line_index': line_index,
                'direction': 'vertical'
            })

    def extend_boundary_line_downwards(self, x, y):
        """向下扩展边界线"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        path = []
        current_x, current_y = x, y
        max_steps = 200  # 限制最大扩展步数
        step_count = 0

        while current_y < binary.shape[0] - 1 and step_count < max_steps:
            path.append((current_x, current_y))

            # 尝试向下移动
            if binary[current_y + 1, current_x] > 0:
                current_y += 1
            # 尝试右下移动
            elif current_x < binary.shape[1] - 1 and binary[current_y + 1, current_x + 1] > 0:
                current_y += 1
                current_x += 1
            # 尝试左下移动
            elif current_x > 0 and binary[current_y + 1, current_x - 1] > 0:
                current_y += 1
                current_x -= 1
            else:
                break

            step_count += 1

        return path, (current_x, current_y)

    def find_template_region_for_point(self, x, y):
        """查找点所在的模板区域"""
        for template_box in self.template_matches:
            tx1, ty1, tx2, ty2 = template_box
            if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                return template_box, self.template_texts.get(template_box, [])

        return None, []

    def associate_texts_with_line(self, line_index, texts):
        """将文本与线段关联"""
        for text in texts:
            self.box_to_line_text_boxes[line_index].setdefault('associated_texts', []).append(text)