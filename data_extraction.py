import random
import scipy.io.wavfile as wav
import python_speech_features as psf
import numpy as np
import speechpy
import itertools
import os.path
from collections import defaultdict
from config import emotions_map


LARGE_WAV_COEF = 300000
FRAME_COEF = 200000
STEP_COEF = 97000


def main():
    wav_files_info = get_speech_files_info()
    train_data = extract_from_wavs_infos(wav_files_info)  # [X, y]
    np.savetxt('data.txt', train_data)
    print train_data[0]


def extract_from_wavs_infos(wav_files_info):
    res = []
    d = {1:None,2:None,3:None,4:None,5:None,6:None,7:None,8:None}
    for wav_file_name, emotion, actor in wav_files_info:
        features_set = extract_from_wav(wav_file_name, 1200, actor)
        res.append(features_set + [emotion])#[emotions_map[int(emotion)]])
        if random.random() < 1/180.:
            d[emotion] = wav_file_name
    print d
    return np.array(res, dtype=np.float64)


def get_speech_files_info():
    # Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    # Vocal channel (01 = speech, 02 = song).
    # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    # Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    # Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    # Repetition (01 = 1st repetition, 02 = 2nd repetition).
    # Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)
    actors = [str(i) if i > 9 else '0' + str(i) for i in xrange(1, 25)]
    modality_set = ['03']
    vocal_channel_set = ['01']
    emotions = [str(em_key) if em_key > 9 else '0' + str(em_key) for em_key in emotions_map.keys()]
    #['03', '05', '08']#
    emotional_intensity_set = ['01', '02']
    statements = ['01', '02']
    repetitions = ['01', '02']

    files_info = []

    for params in itertools.product(actors, modality_set, vocal_channel_set, emotions,
                                    emotional_intensity_set, statements, repetitions):
        # actor, modality, vocal_channel, emotional_intensity, statements, repetitions, emotion = params
        actor = params[0]
        emotion = params[3]
        emotional_intensity = params[4]
        if emotion == '01' and emotional_intensity == '02':
            pass  # NOTE: There is no strong intensity for the 'neutral' emotion.
        else:
            file_name = '-'.join(list(params[1:]) + [actor]) + '.wav'
            files_info.append((file_name, emotion, actor))
    return files_info


def extract_from_wav(wav_file_name, nfft, actor=None):
    if actor is not None:
        wav_file_path = os.path.join('speech_data', 'Actor_' + actor, wav_file_name)

    fs, signal = get_signal(wav_file_name)
        
    # return fs, signal
    return extract_from_signal(fs, signal, nfft)


def extract_from_large_wav(wav_file_name, nfft):
    fs, signal = get_signal(wav_file_name)
    if len(signal.shape) == 2:
        signal = (signal[:, 0] + signal[:, 1])/2.
        fs /= 2.
    current_pos = 0
    length = len(signal)
    lst = []
    while current_pos < length:
        s = signal[current_pos: current_pos+FRAME_COEF]
        lst.append(extract_from_signal(fs, s, nfft))
        current_pos += STEP_COEF
    return lst


def get_signal(wav_file_path):
    fs, signal = wav.read(wav_file_path)
    # if len(signal.shape) == 2:
    #     signal = (signal[:, 0] + signal[:, 1])/2.
    #     fs /= 2.
    return fs, signal


def extract_from_signal(fs, signal, nfft):
    mfcc = psf.mfcc(signal, fs, nfft=nfft)
    fbank = psf.fbank(signal, fs, nfft=nfft)[0]
    logfbank = psf.logfbank(signal, fs, nfft=nfft)
    ssc = psf.ssc(signal, fs, nfft=nfft)

    mfcc_mean = [mfcc[:, i].mean() for i in xrange(mfcc.shape[1])]
    mfcc_std = [mfcc[:, i].std() for i in xrange(mfcc.shape[1])]
    fbank_mean = [fbank[:, i].mean() for i in xrange(fbank.shape[1])]
    fbank_std = [fbank[:, i].std() for i in xrange(fbank.shape[1])]
    logfbank_mean = [logfbank[:, i].mean() for i in xrange(logfbank.shape[1])]
    logfbank_std = [logfbank[:, i].std() for i in xrange(logfbank.shape[1])]
    ssc_mean = [ssc[:, i].mean() for i in xrange(ssc.shape[1])]
    ssc_std = [ssc[:, i].std() for i in xrange(ssc.shape[1])]

    return mfcc_mean + mfcc_std + fbank_mean + fbank_std + logfbank_mean + logfbank_std + ssc_mean + ssc_std

    #signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)
    frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                                              filter=lambda x: np.ones((x,)), zero_padding=True)

    power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
    mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                    num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                    num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)

    # power_spectrum_mean = [power_spectrum[:, i].mean() for i in xrange(power_spectrum.shape[1])]
    # power_spectrum_std = [power_spectrum[:, i].std() for i in xrange(power_spectrum.shape[1])]
    mfcc_mean = [mfcc[:, i].mean() for i in xrange(mfcc.shape[1])]
    mfcc_std = [mfcc[:, i].std() for i in xrange(mfcc.shape[1])]
    # logenergy_mean = [logenergy[:, i].mean() for i in xrange(logenergy.shape[1])]
    # logenergy_std = [logenergy[:, i].std() for i in xrange(logenergy.shape[1])]
    return mfcc_mean + mfcc_std
    # return power_spectrum_mean + power_spectrum_std + mfcc_mean + mfcc_std + logenergy_mean + logenergy_std


def is_large(wav_file_name):
    fs, signal = get_signal(wav_file_name)
    # print len(signal), LARGE_WAV_COEF
    return len(signal) > LARGE_WAV_COEF


if __name__ == '__main__':
    main()
