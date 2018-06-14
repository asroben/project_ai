import tensorflow as tf
import numpy as np
import cv2
import os

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

from PIL import Image


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


def train_classifier(embeddings, labels):
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    svm = SVC(probability=True)

    clf = GridSearchCV(svm, param_grid, cv=10, n_jobs=4)

    clf.fit(x_train, y_train)

    print(clf.score(x_test, y_test))

    return clf

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# python2 retrain.py --image_dir "augmented_data(2)" --output_graph "output_graph.pb" --output_labels "output_labels.txt" --how_many_training_steps 200 --train_batch_size 50 --validation_batch_size 50 --architecture="20180402-114759/model-20180402-114759"
# Add layer before input to map to network image size
# Retrain output with fully connected layer to classify
def finetune_cnn(pretrained_model, data, labels):
    number_of_classes=5
    embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
    input_size = int(embeddings.get_shape()[1])
    W = init_weights([input_size, number_of_classes])
    b = init_bias([number_of_classes])
    output_layer = tf.matmul(embeddings, W) + b
    optimizer = tf.train.AdamOptimizer()
    loss = tf.losses.softmax_cross_entropy(labels=labels, logits=output_layer)
    train_op = optimizer.minimize(loss, var_list=[output_layer])
    print("Training...")
    pretrained_model.run(train_op, feed_dict={x: data,
                                              y: labels})
    return


if __name__ == '__main__':
    image_path = "augmented_data/"
    sess = tf.Session()
    pretrained_model = load_model(sess)
    image_data, labels = load_dataset(image_path)
    finetune_cnn(pretrained_model, image_data["train"], labels["train"])
    # embeddings = embed_images(image_data["train"])
    # clf = train_classifier(embeddings, labels["train"])