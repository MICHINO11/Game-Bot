# Arda Mavi
import os
import numpy as np
from keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split

def imresize(img, size):
    """
    Reemplazo de scipy.misc.imresize usando Pillow.
    size: tuple (height, width, channels) o (height, width)
    """
    if len(size) == 3:
        size = (size[0], size[1])
    return np.array(Image.fromarray(img).resize(size, Image.BICUBIC))

def imread(path):
    """Leer imagen usando Pillow"""
    img = Image.open(path).convert('RGB')
    return np.array(img)

def imsave(path, img):
    """Guardar imagen usando Pillow"""
    Image.fromarray(img).save(path)

def get_img(data_path):
    img = imread(data_path)
    img = imresize(img, (150, 150, 3))
    return img

def save_img(img, path):
    imsave(path, img)
    return

def get_dataset(dataset_path='Data/Train_Data'):
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except:
        labels = os.listdir(dataset_path)  # Obtener etiquetas
        X = []
        Y = []
        count_categori = [-1, '']  # Para codificar etiquetas
        for label in labels:
            datas_path = dataset_path + '/' + label
            for data in os.listdir(datas_path):
                img = get_img(datas_path + '/' + data)
                X.append(img)
                # Para codificar etiquetas:
                if data != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = data.split(',')
                Y.append(count_categori[0])

        # Crear dataset
        X = np.array(X).astype('float32') / 255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, count_categori[0] + 1)

        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test

