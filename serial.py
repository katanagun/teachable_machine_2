import tensorflow as tf
import numpy as np
import PIL
import time
import os

# Загрузка TFLite модели
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\bogdan\PycharmProjects\teachable_machine\converted_tflite\model_unquant.tflite")
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Путь к папке с изображениями
image_dir = r"C:\Users\bogdan\PycharmProjects\teachable_machine\images"

# Проверка существования папки
if not os.path.exists(image_dir):
    print(f"Ошибка: Папка с изображениями '{image_dir}' не найдена.")
    exit()

# Список файлов изображений в папке
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# Определение меток
labels = ['Like', 'Dislike']

all_execution_time = 0

for image_path in image_paths:
    try:
        start_time = time.time()
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224), color_mode='rgb')
        input_data = tf.keras.preprocessing.image.img_to_array(image)
        input_data = input_data.reshape(1, 224, 224, 3)
        input_data = input_data / 255.0  # Нормализация

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        end_time = time.time()
        execution_time = end_time - start_time
        all_execution_time += execution_time
        predicted_class_index = np.argmax(output_data, axis=1)
        predicted_class_label = labels[predicted_class_index[0]]

        print(f"Обработка изображения: {image_path}")
        print("Время выполнения:", execution_time, "секунд")
        print("Выходные данные модели:", output_data)
        print("Предсказанный класс:", predicted_class_label)
        print("-" * 20)  # Разделитель между изображениями
        print("Общее вреся выполнения: ", all_execution_time)

    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")