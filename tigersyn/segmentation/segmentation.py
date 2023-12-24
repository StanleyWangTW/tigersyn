import os
from os.path import basename, join
import glob
import time

import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np

from . import syn_tool

all_model = dict()
all_model['syn'] = r'any_synthseg_v003_synthseg.onnx'
all_model['hippo'] = r'any_hippocampus_v001_mix.onnx'


def produce_mask(model, f, GPU):
    model_ff = syn_tool.get_model(model)
    input_nib = nib.load(f)
    input_nib_resp = syn_tool.read_file(model_ff, f)

    mask_nib_resp = syn_tool.run(model_ff, input_nib_resp, GPU=GPU)

    mask_nib = resample_to_img(mask_nib_resp, input_nib, interpolation="nearest")

    output = mask_nib.get_fdata()

    if np.max(output) <= 255:
        dtype = np.uint8
    else:
        dtype = np.int16

    output = output.astype(dtype)

    output_nib = nib.Nifti1Image(output, input_nib.affine, input_nib.header)
    output_nib.header.set_data_dtype(dtype)

    return output_nib


def run(argstring, input, output=None, model=None):
    syn = "s" in argstring
    hippo = 'h' in argstring
    gpu = "g" in argstring
    get_z = 'z' in argstring

    if not isinstance(input, list):
        input = [input]

    input_file_list = input
    if os.path.isdir(input[0]):
        input_file_list = glob.glob(join(input[0], '*.nii'))
        input_file_list += glob.glob(join(input[0], '*.nii.gz'))

    elif '*' in input[0]:
        input_file_list = glob.glob(input[0])

    output_dir = output

    print('Total nii files:', len(input_file_list))
    count = 0
    for f in input_file_list:
        count += 1
        t = time.time()

        f_output_dir = output_dir
        if f_output_dir is None:
            f_output_dir = os.path.dirname(os.path.abspath(f))
        else:
            os.makedirs(f_output_dir, exist_ok=True)

        print(f'{count} Processing :', os.path.basename(f))

        ftemplate = basename(f).replace('.nii', '_@@@@.nii')
        if get_z and '.gz' not in ftemplate:
            ftemplate += '.gz'
        ftemplate = join(f_output_dir, ftemplate)

        if syn:
            aseg_nib = produce_mask(all_model['syn'], f, GPU=gpu)
            fn = syn_tool.save_nib(aseg_nib, ftemplate, 'syn')

        if hippo:
            hippo_nib = produce_mask(all_model['hippo'], f, GPU=gpu)
            fn = syn_tool.save_nib(hippo_nib, ftemplate, 'hippocampus')

        print('Processing time: %d seconds' % (time.time() - t))


if __name__ == "__main__":
    pass
