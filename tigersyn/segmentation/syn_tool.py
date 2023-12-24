import os
from os.path import abspath, basename, isfile, join, dirname
import sys
import warnings

import nibabel as nib
from nilearn.image import reorder_img, resample_img
import numpy as np
import onnxruntime as ort

warnings.filterwarnings("ignore", category=UserWarning)

label_all = dict()
label_all['synthseg'] = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42,
                         43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)

model_servers = ['https://github.com/StanleyWangTW/tigersyn/releases/download/modelhub/']

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = dirname(sys.executable)
elif __file__:
    application_path = dirname(dirname(abspath(__file__)))

model_path = join(application_path, 'models')
os.makedirs(model_path, exist_ok=True)


def get_model(f):
    if isfile(f):
        return f

    if '.onnx' in f:
        fn = f
    else:
        fn = f + '.onnx'

    model_file = join(model_path, fn)

    if not os.path.exists(model_file):
        raise RuntimeError(f'Model not found. Should at {model_file}')

    return model_file


def read_nib(input_nib):
    """nib to 5D numpy"""
    return input_nib.get_fdata()[None, None, ...]


def save_nib(data_nib, ftemplate, postfix):
    output_file = ftemplate.replace('@@@@', postfix)
    nib.save(data_nib, output_file)
    print('Writing output file: ', output_file)
    return output_file


def read_file(model_ff, input_file):
    """load nib and reorder"""
    input_nib = nib.load(input_file)
    vol_nib = reorder_img(input_nib, resample="continuous")

    if model_ff.split('_')[1] == 'synthseg':
        vol_nib = resample_voxel(vol_nib, (1, 1, 1), interpolation='continuous')

    return vol_nib


def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]  # aseg43, bet
    # print(seg_mode, version , model_str)

    return seg_mode, version, model_str


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def resample_voxel(data_nib, voxelsize, target_shape=None, interpolation='continuous'):
    affine = data_nib.affine
    target_affine = affine.copy()

    factor = np.zeros(3)
    for i in range(3):
        factor[i] = voxelsize[i] / \
            np.sqrt(affine[0, i]**2 + affine[1, i]**2 + affine[2, i]**2)
        target_affine[:3, i] = target_affine[:3, i] * factor[i]

    new_nib = resample_img(data_nib,
                           target_affine=target_affine,
                           target_shape=target_shape,
                           interpolation=interpolation)

    return new_nib


def predict(model, data, GPU):
    """ model: onnx file path
        data: numpy.Array
        GPU: bool
        read data, then segmentation"""
    if GPU and (ort.get_device() == "GPU"):
        ort_sess = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
    else:
        ort_sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    data_type = 'float32'
    return ort_sess.run(None, {'input': data.astype(data_type)})[0]


def run(model_ff, input_nib, GPU):
    """segment nib, return mask_nib"""
    seg_mode, _, model_str = get_mode(model_ff)

    data = read_nib(input_nib)
    # data = normalize(data)

    logits = predict(model_ff, data, GPU)[0, ...]

    if seg_mode in ['synthseg']:
        mask_pred = np.argmax(logits, axis=0)

        labels = label_all[seg_mode]
        mask_pred_relabel = mask_pred * 0
        for ii in range(len(labels)):
            mask_pred_relabel[mask_pred == (ii + 1)] = labels[ii]
            # print((ii+1), labels[ii])
        mask_pred = mask_pred_relabel

    if seg_mode in ['hippocampus']:

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        mask_pred = sigmoid(logits[0, ...])
        mask_pred[mask_pred >= 0.5] = 1
        mask_pred[mask_pred < 0.5] = 0

    mask_pred = mask_pred.astype(int)
    output_nib = nib.Nifti1Image(mask_pred, input_nib.affine, input_nib.header)

    return output_nib
