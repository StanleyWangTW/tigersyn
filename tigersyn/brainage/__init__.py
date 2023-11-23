from predict_brainage import predict_age

if __name__ == '__main__':
    f = r'tigersyn\brainage\tests\data\CC0001_philips_15_55_M_aseg.nii.gz'
    model = r'tigersyn\models\aseg_age_v001_linreg.onnx'
    print(predict_age(f, model))
