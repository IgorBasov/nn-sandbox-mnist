#! /usr/bin/env python
# -*- coding: utf-8 -*-

# подключаем набор тренировочных данных
from tensorflow.keras.datasets import fashion_mnist
# подключаем модель сети, в которой слои следуют последовательно (друг за другом)
from tensorflow.keras.models import Sequential
# нейронная сеть будет полносвязной. В Keras такие сети называются Dense
from tensorflow.keras.layers import Dense
# утилиты Keras для преобразования формата данных
from tensorflow.keras import utils

import numpy as np

# ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ

# загружаем данные при помощи встроенных в Keras инструментов
# x_train - изображения для обучения
# y_train - правильные ответы для обучения
# x_test - изображения для проверки
# y_test - правильные ответы для проверки
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# преобразование размерности изображений
# 60000 изображений по 28*28 пикселей нужно преобразовать
# в массив из 60000 одномерных массивов длиной 784 элемента каждый
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# нормализация данных:
# делим интенсивность каждого пикселя на 255, чтобы получить значения
# в отрезке [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# ПОДГОТОВКА ПРАВИЛЬНЫХ ОТВЕТОВ

# преобразование меток в категории
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# названия классов
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

# СОЗДАНИЕ НЕЙРОННОЙ СЕТИ

# создаем последовательную модель (в Keras нейронные сети называются моделями)
# последовательная модель: слои друг за другом
model = Sequential()

# добавляем уровни сети (слои)
# входной слой:
# 800 - количество нейронов
# input_dim=784 - количество входов в каждый нейрон
# activation="relu" - функция активации

# 784-2500-2000-1500-1000-500-10

firstLayer = Dense(2500, input_dim=784, activation="relu")
model.add(firstLayer)
# model.add(Dense(2000, activation="relu"))
# model.add(Dense(1500, activation="relu"))
# model.add(Dense(1000, activation="relu"))
# model.add(Dense(500, activation="relu"))

# выходной слой
# 10 - количество нейронов
# input_dim - можно не указывать. По умолчанию равно количесту нейронов предыдущего слоя
# activation="softmax" - функция активации
secondLayer = Dense(10, activation="softmax")
model.add(secondLayer)

# компилируем модель
# loss="categorical_crossentropy" - функция ошибки. categorical_crossentropy - категориальная перекрестная энтропия (вместо среднеквадратичного отклонения)
# categorical_crossentropy - хорошо подходит для задач классификации, если классов боль, чем 2
# optimizer="SGD" - тип оптимизатора SGD - стохастический градиентный спуск
# metrics=["accuracy"] - метрики качества обучения
# accuracy - доля правильных ответов нейронной сети
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])


# ОБУЧЕНИЕ СЕТИ

# для обучения исп. метод стохастичестого градиентного спуска с минивыборками
model.fit(
  x_train, # данные
  y_train, # ответы
  batch_size=200, # размер минивыборки
  epochs=100, # количество эпох обучения (одна эпоха - обучение НС на всем наборе данных)
  verbose=1 # печать прогресса обучения НС
)

# РАСПОЗНАВАНИЕ ПРЕДМЕТА ОДЕЖДЫ

# запуск сети на входных данных
predictions = model.predict(x_train)

# вывод одного из результатов распознавания
print(predictions[0])

# вывод номера класса, предсказанного нейросетью
print("Predicted: ")
print(np.argmax(predictions[0]))

# вывод правильного номера класса
print("Actual: ")
print(np.argmax(y_train[0]))

# ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ

scores = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy:", round(scores[1] * 100, 4), "%")