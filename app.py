# Импортировать необходимые библиотеки
from flask import Flask, render_template, request, jsonify

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Загрузить модель
model = load_model("model/v3_pred_cott_dis.h5")
print('@@ Model loaded')


def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(150, 150))  # Загружаем лист
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # преобразовать изображение в массив np и нормализовать
    test_image = np.expand_dims(test_image, axis=0)  # изменить размер 3D на 4D

    result = model.predict(test_image).round(3)  # предсказать больное растение или нет
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # получить индекс максимального значения
    print(pred)

    if pred == 0:
        return "Cгорел хлопчатник", 'healthy_plant_leaf.html'  # если индекс 0 сгорел лист
    elif pred == 1:
        return 'Больной хлопчатник', 'disease_plant.html'  # # если index 1
    elif pred == 2:
        return 'Здоровый лист хлопчатник', 'healthy_plant.html'  # если индекс 2 свежий лист
    else:
        return "Здоровый хлопчатник", 'healthy_plant.html'  # если index 3


# Создать экземпляр Flask
app = Flask(__name__)


# отобразить страницу index.html
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# получить входное изображение от клиента, затем предсказать класс и отобразить соответствующую страницу .html для решения
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # ввод данных
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


@app.route("/api/predict", methods=['POST'])
def api_predict():
    if request.method == 'POST':
        file = request.files['image']  # ввод данных
        filename = file.filename
        file_path = os.path.join('static/user_uploaded', filename)
        file.save(file_path)
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
        return jsonify({"prediction": pred})


# Для локальной системы и облака
if __name__ == "__main__":
    app.run(threaded=False, )
