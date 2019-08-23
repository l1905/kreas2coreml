from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model

import math


# dimensions of our images.
# img_width, img_height = 150, 150
# single_data_dir = '/Users/litong/codesrc/python-demo/keras/data/single'
# batch_size = 20
#
# model = load_model('/Users/litong/codesrc/python-demo/keras/my_model.h5')
#
# #加载图片
# fname = "/Users/litong/codesrc/python-demo/keras/data/single/" + "dog.1114.jpg"
# image = np.array(plt.imread(fname))
# print(image.shape)
# image = image.reshape(1, 150, 150, 3)
#
# prediction = model.predict(image)
# print(prediction)

single_data_dir = '/Users/litong/codesrc/python-demo/keras/data/single/'
img_width, img_height = 150, 150
batch_size = 20
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


single_generator = train_datagen.flow_from_directory(
    single_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical') #多分类

model = load_model('/Users/litong/codesrc/python-demo/keras/my_model.h5')

filenames = single_generator.filenames
nb_samples = len(filenames)
print(nb_samples)
prediction_list = model.predict_generator(single_generator,steps = math.ceil(nb_samples/batch_size))

new_prediction_list = []

print(math.ceil(nb_samples/batch_size))
print(len(prediction_list))

#todo 这里需要确认为啥不能遍历出来， 即最里面结构体不是index 需要最终确认
y_pred = prediction_list > 0.5
print(y_pred)
# for i in range(len(prediction_list)):
#     print(prediction_list[i])
    # new_prediction_list[i][0] = "{0:0.4f}".format(new_prediction_list[i][0])
    # new_prediction_list[i][1] = "{0:0.4f}".format(new_prediction_list[i][1])

# print(new_prediction_list)