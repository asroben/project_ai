import tensorflow as tf
import numpy as np
import cv2
import os

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

from PIL import Image
from resizeimage import resizeimage


def load_model(sess):
    saver = tf.train.import_meta_graph('20180402-114759/model-20180402-114759.meta')
    saver.restore(sess, '20180402-114759/model-20180402-114759.ckpt-275')
    return saver

def retrieve_image_data(image_data, labels, key, path):
    for root, dirs, files in os.walk(path):
        if dirs:
            label_dict = {k:v for v, k in enumerate(dirs)}
        else:
            label = label_dict[root.split('/')[-1]]
        for file in files:
            im = resize_image(cv2.imread(root + "/" + file))
            image_data[key].append(im)
            labels[key].append(label)
    return image_data, labels


def load_dataset(path):
    val_dir = path + "val/"
    image_data = {"val": [], "train": []}
    labels = {"val": [], "train": []}
    train_dir = path + "train/"
    image_data, labels = retrieve_image_data(image_data, labels, "train", train_dir)
    image_data, labels = retrieve_image_data(image_data, labels, "val", val_dir)
    return image_data, labels


def resize_image(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im, mode='RGB')
    im = im.resize((160, 160))
    im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    return im


def embed_images(image_list):
    embeddings = sess.run('embeddings:0',
                      feed_dict={'input:0': image_list, 'phase_train:0': False})
    return embeddings


def train_classifier(embeddings, labels):
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    svm = SVC(probability=True)

    clf = GridSearchCV(svm, param_grid, cv=2, n_jobs=4)

    clf.fit(x_train, y_train)

    print(clf.score(x_test, y_test))

    return clf


def finetune_cnn():
    return


if __name__ == '__main__':
    image_path = "augmented_data/"
    sess = tf.Session()
    pretrained_model = load_model(sess)
    image_data, labels = load_dataset(image_path)
    embeddings = embed_images(image_data["train"])
    clf = train_classifier(embeddings, labels["train"])