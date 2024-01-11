# *_*coding:utf-8 *_*
import os
import glob
import time
import subprocess
import concurrent.futures
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil


# OpenFace WIKI: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
"""
-simscale <float> scale of the face for similarity alignment (default 0.7)
-simsize <int> width and height of image in pixels when similarity aligned (default 112)
-format_aligned <format> output image format for aligned faces (e.g. png or jpg), any format supported by OpenCV
-format_vis_image <format> output image format for visualized images (e.g. png or jpg), any format supported by OpenCV. Only applicable to FaceLandmarkImg
-nomask forces the aligned face output images to not be masked out
-g output images should be grayscale (for saving space)

-pose output head pose (location and rotation)
-aus output the Facial Action Units
-gaze output gaze and related features (2D and 3D locations of eye landmarks)
-hogalign output extracted HOG feaure file
-simalign output similarity aligned images of the tracked faces
-nobadaligned if outputting similarity aligned images, do not output from frames where detection failed or is unreliable (thus saving some disk space)    
"""
OPENFACE_EXE = './OpenFace_2.2.0_win_x64/FeatureExtraction.exe'

def process_one_video(video_file, in_dir, out_dir, openface_exe=OPENFACE_EXE, img_size=112):
    # file_name = os.path.basename(os.path.splitext(video_file)[0])
    # Note: + '\\' is needed
    file_name = os.path.splitext(video_file.replace(in_dir + '\\', ''))[0] # out dir has the same structure with in dir
    out_dir = os.path.join(out_dir, file_name)
    if os.path.exists(out_dir):
        print(f'Note: "{out_dir}" already exist!')
        return video_file
    else:
        os.makedirs(out_dir)

    cmd = f'"{openface_exe}" -f "{video_file}" -out_dir "{out_dir}" -simalign -simsize {img_size} -format_aligned jpg -nomask'
    print(cmd)
    subprocess.call(cmd, shell=False)
    return video_file

def main(video_dir, out_dir, openface_exe=OPENFACE_EXE, multi_process=True, video_template_path='*.mp4', img_size=112):
    #===============Need to be modified for different dataset===============
    video_files = glob.glob(os.path.join(video_dir, video_template_path))  # with emo dir

    n_files = len(video_files)
    print(f'Total videos: {n_files}.')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    start_time = time.time()
    if multi_process:
        #using multi process to extract acoustic features for each video file
        # count = 0
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     tasks = [executor.submit(process_one_video, video_file, video_dir, out_dir, openface_exe) \
        #              for video_file in video_files]
        #     for task in concurrent.futures.as_completed(tasks):
        #         try:
        #             video_file = task.result()
        #             count += 1
        #         except Exception as e:
        #             print('When process "{}", exception "{}" occurred!'.format(video_file, e))
        #         else:
        #             print(f'\t"{video_file:<50}" done, rate of progress: {100.0 * count / n_files:3.0f}% ({count}/{n_files})')
        Parallel(n_jobs=8)(delayed(process_one_video)(video_file, video_dir, out_dir, openface_exe, img_size) \
                     for video_file in tqdm(video_files))
    else:
        for i, video_file in enumerate(video_files, 1):
            print(f'Processing "{os.path.basename(video_file)}"...')
            process_one_video(video_file, video_dir, out_dir, openface_exe, img_size)
            print(f'"{os.path.basename(video_file)}" done, rate of progress: {100.0 * i / n_files:3.0f}% ({i}/{n_files})')

    end_time = time.time()
    print('Time used for video face extraction: {:.1f} s'.format(end_time - start_time))

def copy_one_video(src_dir, tgt_dir):
    shutil.copytree(src_dir, tgt_dir)
    print(f'Copy "{src_dir}" to "{tgt_dir}"')


if __name__ == '__main__':
    # CAMER-D dataset (downloaded from: https://github.com/CheyneyComputerScience/CREMA-D)
    dataset_root = 'path/to/CREMA-D'
    video_dir = os.path.join(dataset_root, 'VideoFlash') # note: .avi
    img_size = 256
    file_ext = 'flv'
    video_template_path = f'*.{file_ext}'
    out_dir = os.path.join(video_dir, '../openface')
    # STEP 1: extract faces from videos using OpenFace
    main(video_dir, out_dir, video_template_path=video_template_path, multi_process=True, img_size=img_size)

    # STEP 2: reorganize the extracted face directories
    src_root = out_dir
    tgt_root = out_dir.replace('openface', 'face_aligned') # in fact, reorganized
    count = 0
    src_dirs, tgt_dirs = [], []
    for sample_dir in os.scandir(src_root):
            sample_name = sample_dir.name
            tgt_dir = os.path.join(tgt_root, sample_name) # organize videos in the original way
            src_dir = os.path.join(sample_dir, f'{sample_name}_aligned')
            src_dirs.append(src_dir)
            tgt_dirs.append(tgt_dir)
            count += 1
    print(f'Total videos: {count}.')
    Parallel(n_jobs=16)(delayed(copy_one_video)(src_dir, tgt_dir) \
                       for src_dir, tgt_dir in tqdm(zip(src_dirs, tgt_dirs)))

