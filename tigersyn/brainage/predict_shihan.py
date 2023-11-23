import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import tqdm
import joblib

from utils import get_volumes

labels = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43,
    44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
]
# mean & std for normalization
u = joblib.load('u_values.joblib')
s = joblib.load('s_values.joblib')


def predict_age(f):
    # 將腦組織體積大小作為特徵
    x_test = get_volumes(f, labels)
    x_test = np.expand_dims(x_test, axis=0)
    x_test_normalized = (x_test - u) / s

    # 2: 特徵選擇
    # 載入 ONNX 張量
    selected_features_onnx = onnx.load_tensor('selected_features.onnx')
    selected_features_array = numpy_helper.to_array(selected_features_onnx)
    selected_features_loaded = selected_features_array.tolist()

    # 使用載入的特徵進行後續處理
    x_test_selected = x_test_normalized[:, selected_features_loaded]

    # 載入 ONNX 模型
    onnx_model_path = "custom_keras_regressor.onnx"
    onnx_session = ort.InferenceSession(onnx_model_path,
                                        providers=['CUDAExecutionProvider'])

    # 7. 模型評估
    input_data = np.array(x_test_selected, dtype=np.float32)
    onnx_output = onnx_session.run(None, {'input_node': input_data})

    # 取得 ONNX 模型的預測結果
    y_test_pred_onnx = np.squeeze(onnx_output[0])
    # 找到最可能的年齡
    predicted_age = np.mean(y_test_pred_onnx)
    # 印出最可能的年齡
    # print(f"Predicted Age: {predicted_age} years")
    return predicted_age


if __name__ == '__main__':
    df = pd.read_excel(r'IXI.xls', index_col=0)
    test_files = glob.glob(
        r'D:\Python_Projects\MRI_course\HW3\IXI_aseg\*.nii.gz')
    test_files = test_files[:10]
    print(len(test_files))

    true_ages = list()
    pred_ages = list()
    for f in tqdm.tqdm(test_files):
        IXI_ID = int(os.path.basename(f).split('-')[0].replace('IXI', ''))
        if IXI_ID in df.index:
            if len(df.at[IXI_ID, 'DATE_AVAILABLE'].shape) == 0:
                date_available = df.at[IXI_ID, 'DATE_AVAILABLE']
            else:
                date_available = 0

            if date_available == 1 and not np.isnan(df.at[IXI_ID, 'AGE']):
                true_ages.append(df.at[IXI_ID, 'AGE'])
                pred_ages.append(predict_age(f))

    pred_ages = np.array(pred_ages)
    true_ages = np.array(true_ages)
    MAE = mean_absolute_error(true_ages, pred_ages)

    print(MAE)
