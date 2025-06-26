from paddleocr import LayoutDetection
import fitz
import numpy as np
from PIL import Image
import os
import cv2
import json
# this snippet is used to extract tables from a PDF document using PaddleOCR's LayoutDetection model.
model = LayoutDetection(model_name="PP-DocLayout_plus-L")
pdf_path = "/ssddata/a/projectss/formsimages/CrescentGrowthII-Dec_2024.pdf"
output_dir = "./output/cut_tables/"
os.makedirs(output_dir, exist_ok=True)
output = model.predict(pdf_path, batch_size=1, layout_nms=True)
pages=[]
for line in output:
    pages.append(line)
for page_idx, page in enumerate(pages):
    img = page['input_img']  # 页面图像（NumPy 数组）
    boxes = page['boxes']    # 页面中的检测框

    for box_idx, box in enumerate(boxes):
        if box['label'] == 'table':  # 检查是否为表格
            # 获取表格坐标 [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = map(int, box['coordinate'])

            # 裁剪图像
            table_img = img[y_min:y_max, x_min:x_max]
            output_path = os.path.join(output_dir, f"page_{page_idx}_table_{box_idx}.png")
            cv2.imwrite(output_path, table_img)
            print(f"Saved table at: {output_path}")

