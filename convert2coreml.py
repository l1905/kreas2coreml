import coremltools

# coreml_model = coremltools.converters.caffe.convert('my_caffe_model.caffemodel')
model_path = '/Users/litong/codesrc/python-demo/keras/my_model.h5'

ml_model_path = "/Users/litong/codesrc/python-demo/keras/my_ml_model.mlmodel"

#转化模型
coreml_model = coremltools.converters.keras.convert(model_path, image_input_names="heima")

#保存
# coremltools.utils.save_spec(coreml_model, ml_model_path)
coreml_model.save(ml_model_path)