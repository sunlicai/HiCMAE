"""
注意：该文件最初在preprocess下，且命名为extract_audio_from_video.py，并且已推送至170和164服务器.
"""
import subprocess
import os
import glob

dataset = "MAFW"
data_path = os.path.expanduser(f'~/AC/Dataset/{dataset}')
video_dir = os.path.join(data_path, 'data/clips')
# save dir
audio_dir = os.path.join(data_path, 'data/audio_16k')
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

audio_sample_rate = 16000
audio_file_ext = 'wav'

num_samples = 10045
video_files = sorted(glob.glob(os.path.join(video_dir, f'*.mp4')))
assert len(video_files) == num_samples, f"Error: wrong number of videos, expected {num_samples}, got {len(video_files)}!"
for video_file in video_files:
    sample_name = os.path.basename(os.path.splitext(video_file)[0])
    audio_file = os.path.join(audio_dir, f'{sample_name}.{audio_file_ext}')
    # using FFmpeg to extract audio from video
    subprocess.call(['ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ac', str(1), '-ar', str(audio_sample_rate), audio_file])
    print('{} is done!'.format(sample_name))
print('All Done!')
