import tensorflow as tf
import numpy as np
import PIL

# Загрузка TFLite модели
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\bogdan\PycharmProjects\teachable_machine\converted_tflite\model_unquant.tflite")
interpreter.allocate_tensors()

# Получение информации о входных и выходных тензорах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Загрузка и подготовка входного изображения
image_path = r"C:\Users\bogdan\PycharmProjects\teachable_machine\Like.jpg"
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
print("Выходные данные модели:", output_data)

# Определение меток
labels = ['Like', 'Dislike']

# Получение предсказанного класса
predicted_class_index = np.argmax(output_data, axis=1)
predicted_class_label = labels[predicted_class_index[0]]

print("Предсказанный класс:", predicted_class_label)
