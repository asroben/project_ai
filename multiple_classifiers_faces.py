import tensorflow as tf
import numpy as np
import cv2
import os
import time

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB


from PIL import Image
CLASSIFIERS = ["svm", "bernnb", "dtree", "sgd"]

# Load in training and test set images
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
            # cv2.imwrite("augmented_data(2)/" + "/".join(root.split("/")[2:]) + "/" + file, im)
            image_data[key].append(im)
            labels[key].append(label)
    return image_data, labels


def load_dataset(path):
    val_dir = path + "val/"
    train_dir = path + "train/"
    image_data = {"val": [], "train": []}
    labels = {"val": [], "train": []}
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


def train_classifier(embeddings, labels, classifier):
    # x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)
    x_train = embeddings
    y_train = labels
    if classifier == "svm":
        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
        clfs = SVC(probability=True)

    elif classifier == "bernnb":
        param_grid = [
            {'alpha': [0.1, 0.5, 0.8, 1]}]
        clfs =  BernoulliNB()

    elif classifier == "dtree":
        clfs = DecisionTreeClassifier(random_state=0)
        param_grid = {'criterion': ['gini', 'entropy'],
                     'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
    elif classifier == "sgd":
        param_grid = {
            'alpha': (0.00001, 0.000001),
            'penalty': ('l2', 'elasticnet'),
            'max_iter': (10, 50, 80, 150, 400, 800),
        }
        clfs = SGDClassifier()

    clf = GridSearchCV(clfs, param_grid, cv=10, n_jobs=4)

    print("Training...")
    clf.fit(x_train, y_train)

    return clf

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


if __name__ == '__main__':
    image_path = "augmented_data/"
    sess = tf.Session()
    pretrained_model = load_model(sess)
    image_data, labels = load_dataset(image_path)
    embeddings = embed_images(image_data["train"])
    embeddings_test = embed_images(image_data["val"])
    for classifier in CLASSIFIERS:
        start = time.time()
        clf = train_classifier(embeddings, labels["train"], classifier)
        end = time.time()
        print("Time elapsed: " + str(end-start))
        print(classifier + ":")
        print("Accuracy: " + str(clf.score(embeddings_test, labels["val"])))