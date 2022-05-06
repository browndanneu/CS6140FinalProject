#!/usr/bin/env python
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display


def generate_mel(filepath, dataset):
    y, sample_rate = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.clf()
    librosa.display.specshow(S_DB, sr=sample_rate)
    dir_path = os.path.split(filepath)[0].replace('genres', dataset)
    filename = os.path.split(filepath)[1][:-4]
    plt.savefig('{}/{}.png'.format(dir_path, filename))

def main():
    audio_dir = 'genres/'
    folder_names = ['train/', 'test/', 'valid/']

    for f in folder_names:
        if not os.path.exists(f):
            os.mkdir(f)

    genres = [name for name in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, name))]
    for g in genres:
        print(g)
        src_file_paths = []
        for im in glob.glob(os.path.join(audio_dir, g, '*.wav'), recursive=True):
            src_file_paths.append(im)

        train_files = src_file_paths[0:80]
        valid_files = src_file_paths[80:90]
        test_files = src_file_paths[90:]

        for f in folder_names:
            if not os.path.exists(os.path.join(f + g)):
                os.mkdir(os.path.join(f + g))
        for file in train_files:
            generate_mel(file, 'train')
        for file in valid_files:
            generate_mel(file, 'valid')
        for file in test_files:
            generate_mel(file, 'test')

if __name__ == '__main__':
    main()
