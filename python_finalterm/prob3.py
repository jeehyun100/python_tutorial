# -*- coding: utf-8 -*-
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time


def read_wav(file_name):
    """
    File name으로 wav 파일을 읽는 기능

    Args:
        file_name, String: 카드의 모양, ['Hearts', 'Clubs', 'Diamonds', 'Spades']

    return:
        _rate, int : Sample rate of wav file
        _sig, numpy array : Data read from wav file
    """
    _rate, _sig = read(file_name)
    return _rate, _sig


def plot_wav_helper(axis, rate, sig, index):
    """
    문제 3번에 맞게 wav file plotting 을 도와주는 기능
    x 축은 seconds, y축는 Amplitude 이다

    Args:
        axis, matplotlib.axis :
        rate, int : Sample rate of wav file
        sig, numpy array : Data read from wav file
        index, Integer : The index if subplot

    """
    # subplot을 설정한다.
    _ax = axis[index]
    _ax.set_ylabel('Amplitude')
    _ax.set_xlabel('Seconds')
    time_seconds = np.linspace(0, len(sig) / rate, num=len(sig))
    _ax.plot(time_seconds, source_sig)


def plot_annotator_helper(axis, text, rate, cord_xy, index):
    """
    문제 3번에 맞게 sampling rate,와 playbacktime의 주석 출력을 도와주는 함수

    Args:
        axis, matplotlib.figure :
        text, String : Annotation test
        rate, int : Sample rate of wav file
        cord_xy, tuple : coordinate x, y
        index, Integer : The index if subplot

    """
    _ax = axis[index]
    bbox_props_f = dict(boxstyle='round', fc='yellow', lw=2)
    rate_anno = "{0} : {1}".format(text, rate)
    _ax.annotate(rate_anno, xy=cord_xy, fontsize=10, bbox=bbox_props_f)


def plot_img_helper(axis, index):
    """
    문제 3번에 맞게 sampling rate 설명 문서를 보여주는 기능

    Args:
        axis, matplotlib.axis :
        index, Integer : The index if subplot

    """
    _ax = axis[index]
    _buf = plt.imread('sampling_nyquist.png')
    _ax.imshow(_buf, interpolation='nearest')
    _ax.text(15,1700, '(Please, Open the file sampling_nyquist.png.')
    _ax.set_xticks([])
    _ax.set_yticks([])


if __name__ == "__main__":
    # 전체 plot을 설정한다. 이름을 Problem 3로 한다.
    fig, ax = plt.subplots(2, 1, figsize=(7, 6))
    fig.suptitle('Problem 3')

    # 문제 3.(가)의 답
    # q2.wav를 읽고, 스피커로 출력하고, plotting을 한다.
    # pygame으로 wav 파일을 출력한다.
    # scipy를 통해 읽은 q2 파일의 sampling rate로 초기화 한다.
    # File을 재생하고, 5초 기다린다.
    filename = 'q2.wav'
    source_rate, source_sig = read_wav(filename)
    pygame.mixer.init(source_rate)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    time.sleep(5)
    # wav file을 plotting 한다.
    plot_wav_helper(ax, source_rate, source_sig, 0)

    # 문제 3.(나)의 답
    # sampling rate를 구하고 간단한 설명을 한다
    plot_annotator_helper(ax, "Sampling rate", source_rate, (0.1, -6000), 0)
    plot_img_helper(ax, 1)
    # Sampling rate의 설명은 sampling_nyquist.png 그림파일이나, 아래의 링크로 첨부합니다.
    # https://docs.google.com/presentation/d/1W03LpVHcbUE72VjjSH5HE3GkSPp3-H5ss3JXxNdldcs/edit?usp=sharing

    # 문제 3.(다)의 답
    # sampling rate를 구하고 간단한 설명을 하시고
    duration_seconds = len(source_sig) / float(source_rate)
    plot_annotator_helper(ax, "Playback time", duration_seconds, (2.5, -6000), 0)
    plt.show()
