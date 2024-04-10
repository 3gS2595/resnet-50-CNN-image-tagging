import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from glob import glob
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

IMAGE_SIZE = [224, 224]
trainMyImagesFolder = "C:\\Users\\lucious\\Pictures\\train0"
testMyImagesFolder = "C:\\Users\\lucious\\Pictures\\test0"

myResnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)
print(myResnet.summary())

# freeze the weights
for layer in myResnet.layers:
    layer.trainable = False

Classes = glob('C:\\Users\\lucious\\Pictures\\train0\\*')

print(Classes)
numOfClasses = len(Classes)


# build the model
global_avg_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResnet.output)
PlusFlattenLayer = Flatten()(global_avg_pooling_layer)

# add the last layer
predictionLayer = Dense(numOfClasses, activation='softmax')(PlusFlattenLayer)

model = Model(inputs=myResnet.input, outputs=predictionLayer)
print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory(
    trainMyImagesFolder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    testMyImagesFolder,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

EPOCHS = 200
best_model_file = 'C:\\Users\\lucious\\Desktop\\RESNET\\model0.h5'
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=30, verbose=1)
]

# train
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=callbacks
)

# print the best validation accuracy
best_val_acc = max(r.history['val_accuracy'])
print(f"Best validation Accuracy : {best_val_acc}")

# plot the results / history
plt.plot(r.history['accuracy'], label='Train acc')
plt.plot(r.history['val_accuracy'], label='Val acc')
plt.legend()
plt.show()

plt.plot(r.history['loss'], label='Train loss')
plt.plot(r.history['val_loss'], label='Val loss')
plt.legend()
plt.show()

import tf2onnx

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
