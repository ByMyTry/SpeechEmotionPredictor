import argparse
import warnings
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
#import plotly.plotly as py
from plotly.offline import plot
from plotly.graph_objs import Pie, Scatter

from data_extraction import extract_from_wav, extract_from_large_wav, is_large
from config import emotions_map

LOG_INFO_TAG = '\n[INFO]: '
FRAME_CONST = 4

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pkl', help="choice *.pkl model file")
    parser.add_argument('--wav', type=str, default='test.wav', help="choice *.wav test audio file")

    return parser.parse_args()


def main():
    args = parse_agrs()
    model_file_name = args.model
    wav_file_name = args.wav

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = deserialize_model(model_file_name)
        nfft = 2400
        if is_large(wav_file_name):
            test_wav_f_lst = read_test_wav_f_lst(wav_file_name, nfft)
            a_lst = calc_a_lst(test_wav_f_lst, model)
            wa_lst = calc_wa_lst(a_lst)
            make_lines_plot(wa_lst, wav_file_name)
            print sum(wa_lst), len(wa_lst), sum(wa_lst)/len(wa_lst)
            make_diagram_plot(sum(wa_lst)/len(wa_lst), wav_file_name)
        else:
            test_wav_features = read_test_wav(wav_file_name, nfft)
            proba = predict(model, test_wav_features)
            make_diagram_plot(proba, wav_file_name)


def deserialize_model(model_file_name):
    return joblib.load(model_file_name)


def read_test_wav_f_lst(wav_file_name, nfft):
    wav_f_lst = extract_from_large_wav(wav_file_name, nfft)
    for i in xrange(len(wav_f_lst)):
        wav_f_lst[i] = preprocessing.scale(wav_f_lst[i])
    return wav_f_lst


def calc_a_lst(wav_f_lst, model):
    a_lst = []
    for wav_f in wav_f_lst:
        proba = model.predict_proba(wav_f)[0]
        a_lst.append(proba)
    return a_lst


def calc_wa_lst(a_lst):
    wa_lst = []
    cur_i = FRAME_CONST
    _sum = sum(a_lst[0: cur_i])
    wa_lst.append(_sum / FRAME_CONST)
    while cur_i < len(a_lst):
        _sum -= a_lst[cur_i - FRAME_CONST]
        _sum += a_lst[cur_i]
        wa_lst.append(_sum / FRAME_CONST)
        cur_i += 1
    return wa_lst


def read_test_wav(wav_file_name, nfft):
    x0 = np.array(extract_from_wav(wav_file_name, nfft))
    x0 = preprocessing.scale(x0)
    return x0


def predict(model, test_wav):
    prediction = model.predict(test_wav)
    proba = model.predict_proba(test_wav)[0]
    print proba, prediction, emotions_map[int(prediction)]
    return proba


def make_lines_plot(wa_lst, wav_file_name):
    x = range(0, len(wa_lst))
    np_wa_lst = np.array(wa_lst, dtype=np.float64)
    data = []
    for i in xrange(len(emotions_map.keys())):
        data.append(
            Scatter(x=x, y=np_wa_lst[:, i], mode='lines', name=emotions_map[i+1])
        )
    plot(data, filename='lines-for-' + '_'.join(wav_file_name.split('.')))


def make_diagram_plot(proba, wav_file_name):
    labels, values = [emotions_map[i+1] for i in xrange(len(emotions_map.keys()))], 100 * proba
    print labels, values
    trace = Pie(labels=labels, values=values)
    plot([trace], filename='diagram-for-' + '_'.join(wav_file_name.split('.')))


if __name__ == '__main__':
    main()
