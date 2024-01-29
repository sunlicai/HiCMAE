import os
from PIL import Image

def convert_bmp_to_jpg(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.bmp'):
                file_path = os.path.join(subdir, file)
                image = Image.open(file_path)
                jpg_file = os.path.splitext(file_path)[0] + '.jpg'
                image.save(jpg_file, 'JPEG')
                os.remove(file_path)

# 使用示例
convert_bmp_to_jpg('/data/public_datasets/CREMA-D/face_aligned')
