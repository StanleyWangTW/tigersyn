from os.path import join, dirname, abspath

import joblib
import onnxruntime as ort
import numpy as np

from .utils import get_volumes

labels = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43,
    44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
]

model_dir = join(dirname(dirname(abspath(__file__))), 'models')
model_ff = join(model_dir, 'aseg_age_v001_mlp.onnx')
mean = joblib.load(join(model_dir, 'u_values.joblib'))
std = joblib.load(join(model_dir, 's_values.joblib'))
selected_features_idx = [0, 1, 4, 5, 6, 7, 11, 15, 17, 18, 21, 22, 23, 24, 29]


def normalize(data):
    '''StandardScaler normalization'''
    return (data - mean) / std


def predict_age(f):
    vols = np.expand_dims(get_volumes(f, labels), axis=0)
    vols = normalize(vols)

    selected_vols = vols[:, selected_features_idx]

    sess = ort.InferenceSession(model_ff, providers=['CUDAExecutionProvider'])
    pred_age = sess.run(
        None, {'input_node': selected_vols.astype(np.float32)})[0].item()

    return pred_age


# model_ff = r'tigersyn\models\aseg_age_v001_linreg.onnx'
# selected_features_idx = [1, 2, 6, 10, 15, 18, 19, 23, 25, 29]

# def predict_age(f):
#     vols = np.expand_dims(get_volumes(f, labels), axis=0)
#     selected_vols = vols[:, selected_features_idx].astype(np.float32)

#     sess = ort.InferenceSession(model_ff, providers=["CPUExecutionProvider"])
#     input_name = sess.get_inputs()[0].name
#     output_name = sess.get_outputs()[0].name
#     pred_age = sess.run([output_name], {input_name: selected_vols})[0].item()

#     return pred_age
