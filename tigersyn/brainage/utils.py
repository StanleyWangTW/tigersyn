import numpy as np
import nibabel as nib


def get_volumes(f, labels):
    volumes = np.zeros(len(labels))
    img = nib.load(f)
    data = img.get_fdata()

    voxel_size = np.prod(img.header.get_zooms())
    for idx, lb in enumerate(labels):
        volumes[idx] = np.sum(data == lb) * voxel_size

    return volumes
