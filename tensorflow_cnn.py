# This script uses pixel data to predict handwritten digits using
# a convolutional neural network (CNN) using the 'tensorflow' package.

# This script borrows generously from the Udacity Deep Learning Course, Assignment 4.

################
# IMPORT MODULES
################
# This section loads modules.

# Import modules.
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np

#############
# IMPORT DATA
#############
# This section imports data.

# Import training and test data.
# Working directory must be set to the 'digit_recognizer' repository folder.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

###########
# PREP DATA
###########
# This section preps the data for model training.

# Get training and test data (not labels).
# There are no labels for the test data.
train_data = train.drop('label',
                        axis = 1).as_matrix()
test_data = test.as_matrix()

# Images currently have 1 row per image.
# For a CNN, I need square/rectangle images so the filter can sweep along.
train_data = train_data.reshape(train.shape[0], 28, 28, 1).astype(np.float32)
test_data = test_data.reshape(test.shape[0], 28, 28, 1).astype(np.float32)

# Turn training labels into one-hot vectors.
# Training labels are already integers.
# There are no labels for the test data.
train_labels_one_hot = LabelBinarizer().fit_transform(train.label)

######################
# SET MODEL PARAMETERS
######################
# This section sets model parameters.

# Specify number of classes (output labels).
classes = 10

# Specify image size in pixels.
image_size = 28

# Specify number of channels (1 since the images are greyscale).
channels = 1

# Set # of iterations to run.
steps = 100

# Set batch size.
batch_size = 16

# Set patch size for filter.
patch_size = 5

# Set output depth for filters.
output_depth = 16

# Set number of hidden nodes.
hidden = 64

##############
# CREATE MODEL
##############
# This section creates a CNN using 'tensorflow'.

# Set up graph.
graph = tf.Graph()
with graph.as_default():

    # Set random seed.
    tf.set_random_seed(20170914)

    # Input placeholders for batch processes.
    # Input test data.
    train_place = tf.placeholder(tf.float32,
                                 shape = (batch_size, image_size, image_size, channels))
    train_labels = tf.placeholder(tf.float32,
                                  shape = (batch_size, classes))
    test_data = tf.constant(test_data)

    # Specify convolution filters and biases.
    filter_1 = tf.Variable(tf.truncated_normal([patch_size,
                                                patch_size,
                                                channels,
                                                output_depth],
                                               stddev = 0.1))
    filter_1_bias = tf.Variable(tf.zeros([output_depth]))

    filter_2 = tf.Variable(tf.truncated_normal([patch_size,
                                                patch_size,
                                                output_depth,
                                                output_depth],
                                               stddev = 0.1))
    filter_2_bias = tf.Variable(tf.constant(1.0,
                                            shape = [output_depth]))
    
    # Specify weights and biases for fully connected layers.
    w_1 = tf.Variable(tf.truncated_normal([(image_size // 4) * (image_size // 4) * output_depth,
                                           hidden],
                                          stddev = 0.1))
    b_1 = tf.Variable(tf.constant(1.0,
                                  shape = [hidden]))

    w_2 = tf.Variable(tf.truncated_normal([hidden, classes],
                                          stddev = 0.1))
    b_2 = tf.Variable(tf.constant(1.0,
                                  shape = [classes]))

    # Define training operations.
    # This is only done in a function because the code this script is based on had it this way :p
    def training_computations(data):

        # Specify first convolution.
        conv_1 = tf.nn.conv2d(data,
                              filter_1,
                              [1, 2, 2, 1],
                              padding = 'SAME')

        # Compute 'relu' on 'conv_1'.
        relu_1 = tf.nn.relu(conv_1 + filter_1_bias)

        # Specify second convolution.
        conv_2 = tf.nn.conv2d(relu_1,
                              filter_2,
                              [1, 2, 2, 1],
                              padding = 'SAME')

        # Compute 'relu' on 'conv_2'.
        relu_2 = tf.nn.relu(conv_2 + filter_2_bias)

        # Save dimensions of 'relu_2'.
        shape = relu_2.get_shape().as_list()

        # Reshape 'relu_2' for fully connected layer operations.
        relu_2_reshape = tf.reshape(relu_2,
                                    [shape[0], (shape[1] * shape[2] * shape[3])])

        # Conduct first fully connected layer matrix multiplication and 'relu' operation.
        full_1 = tf.nn.relu(tf.matmul(relu_2_reshape, w_1) + b_1)

        # Return second fully connected layer matrix multiplication.
        return(tf.matmul(full_1, w_2) + b_2)

    # Execute training computation.
    logits = training_computations(train_place)

    # Specify loss.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = train_labels,
                                                                  logits = logits))

    # Specify optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Generate predictions for training and test set.
    train_pred = tf.nn.softmax(logits)
    test_pred = tf.nn.softmax(training_computations(test_data))

# Run graph.
with tf.Session(graph = graph) as session:

    # Initialize variables.
    session.run(tf.global_variables_initializer())

    # Iterate.
    for step in range(steps):

        # Select index to pick batch.
        # Just having the expression before the '%' sign makes intuitive sense.
        # But what if there were so many iterations that the training data ran out??
        # That's what the stuff after the '%' sign does. Why that specific form? No clue.
        cutoff = (step * batch_size) % (train_data.shape[0] - batch_size)

        # Generate batch.
        batch_data = train_data[cutoff : (cutoff + batch_size), :, :, :]
        batch_labels = train_labels_one_hot[cutoff : (cutoff + batch_size)]

        if (step % 20 == 0):
            print(batch_data[1, :, :, :])
            print(batch_labels[1])

        # Run optimizer (defined above) using 'feed_dict' to feed in the batch.
        _, l, pred = session.run([optimizer, loss, train_pred],
                                 feed_dict = {train_place : batch_data,
                                              train_labels : batch_labels})

        # Print progress and metrics.
        if (step % 50 == 0):

            # Print step and loss.
            print('Step ', step)
            print('Loss :', l)
            print('Training Accuracy',
                  tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1),
                                                  tf.argmax(batch_labels, 1)), tf.float32)).eval())

    # There are no test set labels, so I can't print test set accuracy.
    




        
    




