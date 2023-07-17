from os.path import basename

import nibabel as nib
from nilearn.image import reorder_img
import numpy as np
import onnxruntime as ort

label_all = dict()
label_all['syntheseg'] = (
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26,
    28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
)

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
    return vol_nib


def get_mode(model_ff):
    seg_mode, version, model_str = basename(model_ff).split('_')[1:4]  # aseg43, bet
    #print(seg_mode, version , model_str)

    return seg_mode, version, model_str


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def predict(model, data, GPU):
    """read array-like data, then segmentation"""
    if GPU and (ort.get_device() == "GPU"):
        ort_sess = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
    else:
        ort_sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    data_type = 'float32'
    return ort_sess.run(None, {'input': data.astype(data_type)})[0]

def run(model_ff, input_nib, GPU):
    """segment nib, return mask_nib"""
    seg_mode, _ , model_str = get_mode(model_ff)

    data = read_nib(input_nib)
    # data = normalize(data)

    logits = predict(model_ff, data, GPU)[0, ...]
    mask_pred = np.argmax(logits, axis=0)

    if seg_mode in ['syntheseg']:
        labels = label_all[seg_mode]
        mask_pred_relabel = mask_pred * 0
        for ii in range(len(labels)):
            mask_pred_relabel[mask_pred == (ii + 1)] = labels[ii]
            #print((ii+1), labels[ii])
        mask_pred = mask_pred_relabel
    
    mask_pred = mask_pred.astype(int)
    output_nib = nib.Nifti1Image(mask_pred, input_nib.affine, input_nib.header)
    
    return output_nib