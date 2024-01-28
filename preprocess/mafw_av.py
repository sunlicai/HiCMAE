# *_*coding:utf-8 *_*
import os
import pandas as pd
import sys
import glob
import torchaudio
import numpy as np


dataset = "MAFW"
data_path = os.path.expanduser(f'~/AC/Dataset/{dataset}')
split_dir = os.path.join(data_path, 'Train & Test Set/single/no_caption')
video_dir = os.path.join(data_path, 'data/frames')
audio_dir = os.path.join(data_path, 'data/audio_16k')
audio_sample_rate = 16000 # expected
audio_file_ext = 'wav'

num_splits = 5
splits = range(1, num_splits + 1)

# check video
total_samples = 10045
video_dirs = sorted(glob.glob(os.path.join(video_dir, '*')))
assert len(video_dirs) == total_samples, f'Error: wrong number of videos, expected {total_samples}, got {len(video_dirs)}.'

# check audio and its sample rate
audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
total_audios = 10024 # ffmpeg can not extract audios from 21 samples
assert len(audio_files) == total_audios, f'Error: wrong number of audios, expected {total_audios}, got {len(audio_files)}.'
audio_durations = []
for audio_file in audio_files:
    wav, sr = torchaudio.load(audio_file)
    assert sr == audio_sample_rate, f"Error: '{audio_file}' has a sample rate of {sr}, expected {audio_sample_rate}!"
    audio_durations.append(wav.shape[1] / audio_sample_rate)
print(f'Audio duration: mean={np.mean(audio_durations):.1f}s, max={max(audio_durations):.1f}s, min={min(audio_durations):.1f}s.')

# label, 11 single-labeled emotions
labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']
label2idx = {l:idx for idx, l in enumerate(labels)}

# read split file
for split in splits:
    save_dir = f'../saved/data/mafw/audio_visual/single/split0{split}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_split_file = os.path.join(split_dir, f'set_{split}/train.txt')
    df = pd.read_csv(train_split_file, header=None, delimiter=' ')
    train_label_dict = dict(zip(df[0], df[1]))

    test_split_file = os.path.join(split_dir, f'set_{split}/test.txt')
    df = pd.read_csv(test_split_file, header=None, delimiter=' ')
    test_label_dict = dict(zip(df[0], df[1]))

    train_label_list, test_label_list = [], []
    for v, l in train_label_dict.items(): # ex:00025.mp4 anger
        sample_name = v.split('.')[0]
        video_file = os.path.join(video_dir, sample_name)
        audio_file = os.path.join(audio_dir, f"{sample_name}.{audio_file_ext}")
        if not os.path.exists(audio_file):
            print(f"Warning: the audio of sample '{sample_name}' in split {split} training set does not exist, pass it!")
            continue
        label_idx = label2idx[l]
        train_label_list.append([video_file, audio_file, label_idx])
    for v, l in test_label_dict.items(): # ex:00025.mp4 anger
        sample_name = v.split('.')[0]
        video_file = os.path.join(video_dir, sample_name)
        audio_file = os.path.join(audio_dir, f"{sample_name}.{audio_file_ext}")
        if not os.path.exists(audio_file):
            print(f"Warning: the audio of sample '{sample_name}' in split {split} test set does not exist, pass it!")
            continue
        label_idx = label2idx[l]
        test_label_list.append([video_file, audio_file, label_idx])

    total_samples = len(train_label_list) + len(test_label_list)
    print(f'Total samples in split {split}: {total_samples}, train={len(train_label_list)}, test={len(test_label_list)}')

    # write
    new_train_split_file = os.path.join(save_dir, f'train.csv')
    df = pd.DataFrame(train_label_list)
    df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

    new_test_split_file = os.path.join(save_dir, f'test.csv')
    df = pd.DataFrame(test_label_list)
    df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

