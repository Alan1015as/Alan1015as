import sys
from paddleocr import PaddleOCR
import cv2
from matplotlib import pyplot as plt
import os
from paddleocr import DocVLM
def ocr_image_one(image_path,model_name="PP-DocBee2-3B",model_dir="/ssddata/a/projectss/models/PP-DocBee2-3B",output_dir="./output/ocrtest"):
    os.makedirs(model_dir, exist_ok=True)
    model = DocVLM(model_name=model_name, model_dir=model_dir)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    txt_output = os.path.join(output_dir, f"{basename}.txt")
    result = model.predict(
        input={"image": image_path, "query": "识别表格, 以txt格式输出"},
        batch_size=1)
    for res in result:
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(str(res))
        print(f"Processed {image_path}, saved to {txt_output}")
ocr_image_one("/ssddata/a/projectss/output/cut_tables/test2.png", model_name="PP-DocBee2-3B", model_dir="/ssddata/a/projectss/models/PP-DocBee2-3B", output_dir="./output/ocrtest")



