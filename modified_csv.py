import csv

def replace_paths_in_csv(file_path, output_file_path):
    # 旧路径和新路径的部分
    old_path = "/data/ycs/AC/Dataset/MAFW/data/frames"
    new_path = "/data/public_datasets/MAFW/data/frames/frames"

    old_path_2 = "/data/ycs/AC/Dataset/MAFW/data/audio_16k"
    new_path_2 = "/data/public_datasets/MAFW/data/audio"

    # 读取CSV文件
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        lines = [row for row in reader]

    # 替换指定行中的路径
    for row in lines:
        for i, cell in enumerate(row):
            if old_path in cell:
                #print(cell)
                row[i] = cell.replace(old_path, new_path)
                cell = row[i]
                row[i] = cell.replace(old_path_2, new_path_2)
    
    # 将修改后的数据写入新的CSV文件
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)

# 使用函数
for i in range(1, 6, 1):
    for name in 'train','test':
        file_path = f'/home/hao/Project/HiCMAE/saved/data/mafw/audio_visual/single/split0{i}/{name}.csv'  # 输入文件路径
        output_file_path = f'/home/hao/Project/HiCMAE/saved/data/mafw/audio_visual/single/split0{i}/{name}.csv'  # 输出文件路径
        replace_paths_in_csv(file_path, output_file_path)
