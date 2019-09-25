from hparser import *
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step A: get the datasets into signal arrays, like that over here:
# -> get_sigs_as_vecs(data_arr=ds, type=tp)
all_sigs = pd.DataFrame()
all_sigs_y = pd.DataFrame()
classes_no = 0

for ds_name in os.listdir(CSV_DATASETS_PATH):
    print(ds_name)
    if not ds_name.endswith('.csv'):
        continue
    tp = ds2type(ds_name)
    if tp is None:
        continue
    path = os.path.join(CSV_DATASETS_PATH, ds_name)
    # notice that pd thinks the first line is the column name where is in our case we dont have first column names at all - maybe trying fixing it up later or something.
    ds = pd.read_csv(path)
    sigs, ys = get_sigs(df=ds, type=tp)
    print("sigs,ys sizes=", sigs.shape, ys.shape)
    all_sigs = all_sigs.append(sigs)
    all_sigs_y = all_sigs_y.append(ys)
    classes_no += 1

print('classes no is', classes_no)
# Splitting to train/test sets.
x_train, x_test, y_train, y_test = train_test_split(all_sigs, all_sigs_y, test_size=0.2, random_state=0)
x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


######################################################
# First try: using RF trying to predict results.
######################################################
# We will scale over here the data,
# We choose to do it after splitting so no statistical deps will be gained from one another maybe using the following method.
# sc = StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.transform(x_test)

'''
# Running some tester over the data in here, RF probably might be interesting in here.
# I guess that maybe some small net with CNN might be nice somehow also?

regressor = RandomForestClassifier(n_estimators=40, random_state=0)
regressor.fit(X_train, y_train.transpose().iloc[0].ravel())
y_pred = regressor.predict(X_test)

# Printing confusion matrices and other stuff.
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
'''


######################################################
# Another Try: with Deep Learning & CNNs
######################################################

######################################################
# Functions related to building the model
######################################################

import tensorflow as tf
from tensorflow.saved_model import tag_constants
# Training Parameters
learning_rate = 0.0001
num_steps = 2500
num_epochs = 2000 # will run up to the number of steps eventually over here.
batch_size = 32

# Network Parameters
num_input = 3600 # data input shape of a signal with 1 channel avctually is: (3600 x 1)
num_classes = classes_no # total classes no (depends on how many different rates we have analysed beforehand).
dropout = 0.25 # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, for cases of multiple differnet inputs over here.
        x = x_dict['signals']

        # the current data input is a 1-D vector of 3600
        # Reformating here as a [Channel, Time] [Only one channle here but ecg's can have actually more than ones and such.
        # Tensor input become 3-D: [Batch Size, Channel, Time]
        x = tf.reshape(x, shape=[-1, 3600, 1])

        # Convolution Layer with 32 filters and a kernel size of 8
        conv1 = tf.layers.conv1d(x, 48, 10, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling1d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv1d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling1d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

        saver = tf.train.Saver()

    return out


# Define the model function - this uses the conv_net above and generate an EstimatorSpec to learn over above.
# the conv_net returns actually multilayered deep network and we utilize it over here below.
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

######################################################
######################################################

######################################################
# Functions related to saving/reloading the model again.
######################################################

def json_serving_input_fn():
  """Builds the serving inputs."""
  INPUT_COLUMNS = ['signals']
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat] = tf.placeholder(shape=[None], dtype=tf.float64)
  return tf.estimator.export.ServingInpfutReceiver(inputs, inputs)


def save_model(model, export_dir_path=DEFAULT_MODEL_PATH):
    # export the model, this model might be also good for further usages over with json requests for some server.
    full_model_path = model.export_savedmodel(export_dir_base=export_dir_path, serving_input_receiver_fn=json_serving_input_fn,
                                              )
    print('[+] Saved the model generated into %s' % full_model_path)
    return full_model_path

def load_model(full_model_path, sess=None):
    # loads model into a specific session.
    if sess is None:
        sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, ['serve'], full_model_path)
    return sess

def predict_with_sess(sess, input_arr):
    # given a session previously saved with the cool save_model function.
    # we return the predictions array for these inputs.
    # The session should be previously loaded with load_model function above.
    final_res = sess.graph.get_tensor_by_name('ArgMax:0')
    results = [sess.run(final_res, {'Placeholder:0': np.array(inp)}) for inp in input_arr]
    results = np.array(results).transpose()
    return results

######################################################
######################################################

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'signals': x_train}, y=y_train.reshape(-1,),
    batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Save the Model.
save_model(model)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'signals': x_test}, y=y_test.reshape((-1,)),
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
