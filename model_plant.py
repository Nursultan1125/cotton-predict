## Project: Cotton Plant Disease Prediction &amp; Get Cure AI App - IAIP

# Импорт необходимых библиотек
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

keras.__version__

train_data_path = "D:/A_Cotton_Project/cotton_pant_disease_prediction_ai/data/train/"
validation_data_path = "D:/A_Cotton_Project/cotton_pant_disease_prediction_ai/data/val"


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# Это конфигурация дополнения, которую мы будем использовать для обучения
# Он генерирует больше изображений, используя следующие параметры
training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

# Это генератор, который будет читать картинки, найденные в
# в train_data_path и неограниченно генерировать
# Пакет расширенных данных изображения
training_data = training_datagen.flow_from_directory(train_data_path,  # this is the target directory
                                                     target_size=(150, 150),  # all images will be resized to 150x150
                                                     batch_size=32,
                                                     class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

training_data.class_indices

# это конфигурация дополнения, которую мы будем использовать для проверки:
# только масштабирование
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# это аналогичный генератор для данных проверки
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                               target_size=(150, 150),
                                               batch_size=32,
                                               class_mode='binary')

images = [training_data[0][0][0] for i in range(5)]
plotImages(images)

model_path = 'D:/A_Cotton_Project/cotton_pant_disease_prediction_ai/model/v3_red_cott_dis.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Построение сверточной нейронной сети .Получим  cnn model
cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150, 150, 3]),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),  # neural network beulding
    keras.layers.Dense(units=128, activation='relu'),  # input layers
    keras.layers.Dropout(0.1),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=4, activation='softmax')  # output layer
])

# Компилируем cnn model
cnn_model.compile(optimizer=Adam(lr=0.0001),loss='', metrics=['accuracy'])

# Обучим модель
history = cnn_model.fit(training_data,
                        epochs=20,
                        verbose=1,
                        validation_data=valid_data,callbacks=callbacks_list)

# обобщить историю для точности
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# обобщить историю для ошибки
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
history.history