from documentanalyzer import DocumentAnalyzer
import cv2

# 加载图片
image_path = "photo/page_3.png"
image = cv2.imread(image_path)

# 初始化分析器（可选：传入模板图片用于匹配特定区域）
template = cv2.imread("path/to/template.jpg") if template_exists else None
analyzer = DocumentAnalyzer(image=image, template=template, lang='ch')

# 执行分析流程
analyzer.perform_template_matching()
analyzer.detect_lines()
analyzer.filter_and_associate_lines()
analyzer.extend_boundaries_and_associate_text()

# 获取结果
print(f"识别到 {len(analyzer.lines)} 条水平线段")
print(f"识别到 {len(analyzer.template_matches)} 个模板区域")