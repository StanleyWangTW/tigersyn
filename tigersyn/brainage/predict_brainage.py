import onnxruntime as rt
import numpy as np
from utils import get_volumes

labels = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43,
    44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
]
selected_features_idx = [1, 2, 6, 10, 15, 18, 19, 23, 25, 29]


def predict_age(f, model_ff):
    vols = np.expand_dims(get_volumes(f, labels), axis=0)
    selected_vols = vols[:, selected_features_idx].astype(np.float32)

    sess = rt.InferenceSession(model_ff, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_age = sess.run([output_name], {input_name: selected_vols})[0].item()

    return pred_age
