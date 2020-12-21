# Importing libraries
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Model
from keras.layers import *


# Overview of the dataset
images = ['COVID-19 Radiography Database/COVID-19/COVID-19 (1).png',
'COVID-19 Radiography Database/NORMAL/NORMAL (1).png',
'COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (1).png',
'COVID-19 Radiography Database/COVID-19/COVID-19 (50).png',
'COVID-19 Radiography Database/NORMAL/NORMAL (59).png',
'COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (172).png',
'COVID-19 Radiography Database/COVID-19/COVID-19 (120).png',
'COVID-19 Radiography Database/NORMAL/NORMAL (168).png',
'COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (407).png']

fig, axes = plt.subplots(3, 3, figsize=(7, 7))
labels = ['Covid', 'Normal', 'Pneumonia']
i = 0
j = 0

for row in axes:
    for plot in row:
        plot.imshow(cv2.imread(images[j], 0))
        plot.axhline(y=0.5, color='r')
        plot.set_title(labels[i], fontsize=15)
        plot.axis('off')
        i += 1
        j += 1
    i = 0
plt.savefig('./overview2.png')
fig.tight_layout()
plt.show()


# Data Augmentation
transformation_ratio = 0.05

datagen = ImageDataGenerator(rescale=1. / 255,
                             validation_split = 0.2,
                             rotation_range=transformation_ratio,
                             shear_range=transformation_ratio,
                             zoom_range=transformation_ratio,
                             cval=transformation_ratio,
                             horizontal_flip=True,
                             vertical_flip=True)

# load and iterate training dataset
train_it = datagen.flow_from_directory("Data_organized/train", 
                                       target_size=(224,224), 
                                       color_mode='rgb', 
                                       class_mode="categorical",
                                       batch_size=12,
                                       subset = "training")

# Validation Data
val_it = datagen.flow_from_directory("Data_organized/train",
                                     target_size=(224,224),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=12,
                                     subset='validation')

# load and iterate test dataset
test_it = datagen.flow_from_directory("Data_organized/test", 
                                      target_size=(224,224), 
                                      color_mode='rgb', 
                                      class_mode="categorical")

# Importing pre-trained model
width = 224
height = 224
base_model = keras.applications.DenseNet121(
    include_top=False,
    input_shape=(width, height,3))
# Freeze base model
base_model.trainable = False

# Adding Layers to the pre-trained model

# Create inputs with correct shape
inputs = Input(shape=(224, 224, 3))
#call pre-trained model
x = base_model(inputs, training=False)
# Add pooling layer or flatten layer
x = Flatten()(x)
# Add dense layer
x = Dense(256, activation='relu')(x)
# Add a Dropout layer
x = Dropout(.5)(x)
# Add final dense layer
outputs = Dense(3, activation = 'softmax')(x)
# Combine inputs and outputs to create model
model = Model(inputs, outputs)

# Compiling the model
optim = keras.optimizers.Nadam()
model.compile(loss='categorical_crossentropy',
              metrics=["accuracy"],
             optimizer = optim)


# Training the model
history = model.fit_generator(generator = train_it,
                              steps_per_epoch=train_it.samples/train_it.batch_size,
                              epochs=16,
                              validation_data=val_it,
                              validation_steps=test_it.samples/test_it.batch_size,
)

# Evaluating the model
model.evaluate(test_it, steps=test_it.samples/test_it.batch_size)
#plot metrics
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figheight(8)
fig.set_figwidth(16)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(['Acc','Val'], loc = 'upper left')
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(['loss','Val'], loc = 'upper left')
plt.savefig("./evaluation.png")

# save traine model
keras.models.save_model(model, "model.hdf5")