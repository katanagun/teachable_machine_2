import tensorflow as tf
import numpy as np
import os
import time
import threading
from queue import Queue

# Загрузка TFLite модели
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\bogdan\PycharmProjects\teachable_machine\converted_tflite\model_unquant.tflite")
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Путь к папке с изображениями
images_folder_path = r"C:\Users\bogdan\PycharmProjects\teachable_machine\images"

# Получение списка файлов изображений в папке
image_files = [f for f in os.listdir(images_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Определение меток
labels = ['Like', 'Dislike']

def process_image(queue):
    while not queue.empty():
        image_file = queue.get()
        start_time = time.time()  # Начало измерения времени для каждого изображения

        # Загрузка и подготовка входного изображения
        image_path = os.path.join(images_folder_path, image_file)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224), color_mode='rgb')
        input_data = tf.keras.preprocessing.image.img_to_array(image)
        input_data = input_data.reshape(1, 224, 224, 3)
        input_data = input_data / 255.0  # Нормализация

        # Установка входного тензора
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Запуск инференса
        interpreter.invoke()

        # Получение выходного тензора
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Выходные данные модели для {image_file}:", output_data)

        # Получение предсказанного класса
        predicted_class_index = np.argmax(output_data, axis=1)
        predicted_class_label = labels[predicted_class_index[0]]

        end_time = time.time()  # Конец измерения времени для каждого изображения
        elapsed_time = end_time - start_time  # Вычисление времени выполнения для каждого изображения

        print(f"Предсказанный класс для {image_file}:", predicted_class_label)
        print(f"Время выполнения для {image_file}: {elapsed_time:.4f} секунд\n")

        queue.task_done()

# Создание очереди и добавление файлов изображений
queue = Queue()
for image_file in image_files:
    queue.put(image_file)

# Начало измерения общего времени выполнения программы
total_start_time = time.time()

# Создание и запуск потока для обработки изображений
thread = threading.Thread(target=process_image, args=(queue,))
thread.start()

# Ожидание завершения потока
queue.join()

# Конец измерения общего времени выполнения программы
total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

print(f"Общее время выполнения программы: {total_elapsed_time:.4f} секунд")