from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt



batch_size = 16

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(250, 250, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'dataset/val',
        target_size=(250, 250),
        batch_size=batch_size,
        class_mode='categorical')




train_generator = train_datagen.flow_from_directory(
        'dataset/train',  # this is the target directory
        target_size=(250, 250),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

history_1 = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

fig = plt.figure()
fig.set_size_inches(15, 15)
gs = gridspec.GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(history_1.history['loss'])
ax1.plot(history_1.history['val_loss'])
plt.title('Model Loss for vanilla CNN')
ax1.set_ylabel('Loss')

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history_1.history['accuracy'])
ax2.plot(history_1.history['val_accuracy'])
plt.title('Model Accuracy for vanilla CNN')
ax2.set_ylabel('Accuracy')

plt.show()

#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
model.save('model.h5')
