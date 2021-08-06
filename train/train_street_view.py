from __future__ import print_function

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model,load_model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator
# from data.street_view import get_filenames as get_street_view_filenames
import datetime


data_path = os.path.join('data', 'street_view')
# train_filenames, test_filenames = get_street_view_filenames(data_path)
# train_filenames, test_filenames = (os.listdir('/Users/zhongli_mac/OneDrive/project1/image/labeled/train1'), os.listdir('/Users/zhongli_mac/OneDrive/project1/image/labeled/test1'))
# if '.DS_Store' in test_filenames:
#     test_filenames.remove('.DS_Store')
# if '.DS_Store' in train_filenames:
#     train_filenames.remove('.DS_Store')
# train_filenames = [os.path.join('/Users/zhongli_mac/OneDrive/project1/image/labeled/train1', i) for i in train_filenames]
# test_filenames = [os.path.join('/Users/zhongli_mac/OneDrive/project1/image/labeled/test1', i) for i in test_filenames]
# 使用谷歌风景数据
train_filenames= []
for i in range(7):
    root_path = '/Users/zhongli_mac/Downloads/0/0/{}'.format(i)

    for root,dirs,files, in os.walk(root_path):
        for name in files:
            if 'jpg' in name:
                train_filenames.append(os.path.join(root, name))
    # for name in dirs:
    #     print(os.path.join(root, name))
test_filenames= []
root_path = '/Users/zhongli_mac/Downloads/0/0/{}'.format(7,8)
for root, dirs, files, in os.walk(root_path):
    for name in files:
        if 'jpg' in name:
            test_filenames.append(os.path.join(root, name))
print(len(train_filenames), 'train samples')
print(len(test_filenames), 'test samples')
# print(train_filenames)
# print(test_filenames)
# if 'data/street_view/.DS_Store' in train_filenames:
#     train_filenames.remove('data/street_view/.DS_Store')
#     print("删除")
# if 'data/street_view/.DS_Store' in test_filenames:
#     test_filenames.remove('data/street_view/.DS_Store')
#     print("删除")
# model_name = 'rotnet_street_view_resnet50'

# number of classes
nb_classes = 72
# input image shape
input_shape = (224, 224, 3)

# load base model
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)
# model = load_model('../rotnet_street_view_resnet50_keras2.hdf5',
#                    custom_objects={'angle_error': angle_error})
model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.02, momentum=0.9),
              metrics=[angle_error])

# training parameters
batch_size = 32
nb_epoch = 50
output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, str(datetime.datetime.now())+ '.hdf5'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

# training loop

model.fit_generator(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        test_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(test_filenames) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)
