# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys
import glob
import torchaudio
import numpy as np

dataset = "CREMA-D"
data_path = os.path.expanduser(f'./AC/Dataset/{dataset}')
video_dir = os.path.join(data_path, 'face_aligned')
audio_dir = os.path.join(data_path, 'AudioWAV') # already 16k
audio_sample_rate = 16000 # expected
audio_file_ext = 'wav'

num_splits = 5
splits = range(1, num_splits + 1)

# read
total_samples = 7442
num_subjects_per_split = 18 # 91 / 5
sample_dirs = sorted(glob.glob(os.path.join(video_dir, '*')))
assert len(sample_dirs) == total_samples, f'Error: wrong number of videos, expected {total_samples}, got {len(sample_dirs)}.'

# check audio and its sample rate
audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
assert len(audio_files) == total_samples, f'Error: wrong number of audios, expected {total_samples}, got {len(audio_files)}.'
audio_durations = []
for audio_file in audio_files:
    wav, sr = torchaudio.load(audio_file)
    assert sr == audio_sample_rate, f"Error: '{audio_file}' has a sample rate of {sr}, expected {audio_sample_rate}!"
    audio_durations.append(wav.shape[1] / audio_sample_rate)
print(f'Audio duration: mean={np.mean(audio_durations):.1f}s, max={max(audio_durations):.1f}s, min={min(audio_durations):.1f}s.')

labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']
label2idx = {l:idx for idx, l in enumerate(labels)}

# from: https://github.com/CheyneyComputerScience/CREMA-D)
broken_samples = ['1076_MTI_NEU_XX', '1076_MTI_SAD_XX', '1064_TIE_SAD_XX', '1064_IEO_DIS_MD']

for split in splits:
    print(f'Processing split {split} ...')
    save_dir = f'./saved/data/{dataset.lower()}/audio_visual/split0{split}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_label_list, test_label_list = [], []
    for sample_dir in sample_dirs:
        # pass broken videos
        sample_name = os.path.basename(sample_dir)
        if sample_name in broken_samples:
            continue
        sub_id, sentence, label, intensity = sample_dir.split('/')[-1].split('_')
        sub_idx = int(sub_id) - 1001
        label_idx = label2idx[label]
        offset = 0
        if split == num_splits: # for the last split
            offset = 1
        # audio
        audio_file = os.path.join(audio_dir, f'{sample_name}.{audio_file_ext}')
        if not os.path.exists(audio_file):
            print(f"Warning: the audio file '{audio_file}' does not exist, pass it!")
            continue
        if (split-1)*num_subjects_per_split <= sub_idx < split*num_subjects_per_split + offset:
            test_label_list.append([sample_dir, audio_file, label_idx])
        else:
            train_label_list.append([sample_dir, audio_file, label_idx])
    total_samples = len(train_label_list) + len(test_label_list)
    print(f'Total samples: {total_samples}, train={len(train_label_list)}, test={len(test_label_list)}')

    # write
    new_train_split_file = os.path.join(save_dir, f'train.csv')
    df = pd.DataFrame(train_label_list)
    df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

    new_test_split_file = os.path.join(save_dir, f'test.csv')
    df = pd.DataFrame(test_label_list)
    df.to_csv(new_test_split_file, header=None, index=False, sep=' ')


