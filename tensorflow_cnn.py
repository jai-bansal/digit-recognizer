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
train_data = tf.reshape(train_data,
                        [train.shape[0], 28, 28, 1])
test_data = tf.reshape(test_data,
                       [test.shape[0], 28, 28, 1])

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
                                 shape = (batch_size, image_size, image_size, num_channels))




##################
### BASELINE MODEL
###################
### This section implements a baseline random forest model with no feature engineering or modification.
##
### Get training set features.
##train_features = train.drop('label',
##                            axis = 1)
##
### Create and fit random forest classifier.
### This script is mostly instructional so I use a very small number of trees.
### I also only use a subset of 'train' for the random forest to avoid hitting a memory error.
### I assume the order of the data doesn't matter because I just use the first few rows (no random sampling of rows).
##baseline_rf = RandomForestClassifier(n_estimators = 25,
##                                     oob_score = True,
##                                     random_state = 1234)
##baseline_rf.fit(train_features,
##                train['label'])
##
### Generate predictions for training and test data.
##train['baseline_pred'] = baseline_rf.predict(train_features)
##test['baseline_pred'] = baseline_rf.predict(test)
##
### Compute training set accuracy.
### Training set accuracy is 99.9976%.
##100 * sum(train['label'] == train['baseline_pred']) / len(train['baseline_pred'])
##
### Check cross-validation accuracy.
### For random forest, OOB score can be used for cross-validation.
### OOB accuracy is 93.55%.
##print(baseline_rf.oob_score_)
##
### Test set accuracy cannot be checked as I do not have the answers for the test set.
##
### Delete 'baseline_rf' to avoid a memory error later.
##del(baseline_rf)
##
#############
### PCA MODEL
#############
### This section conducts Principal Components Analysis (PCA) before using a random forest model for prediction.
##
### Create and fit principal component analysis object.
### I conduct one PCA removing many of the least useful components.
### I exclude the full component PCA (included in the R branch) to avoid hitting a memory error.
### Normally, I would scale 'train'. But I do not for 2 reasons:
### 1. All of the data is pixel data ranging from 0 to 255 so the scales are identical for all variables.
### 2. Some of the columns are all a single number (I assume 0) which makes scaling fail.
##pca = PCA(n_components = 50,
##          random_state = 1234).fit(train_features.sample(n = 1000))
##
### Apply 'pca' to 'train' and 'test'.
##train_comp = pca.transform(train_features)
##test_comp = pca.transform(test.drop('baseline_pred',
##                                    axis = 1))
##
### Create and fit a random forest using 'train_comp'.
##pca_rf = RandomForestClassifier(n_estimators = 25,
##                                oob_score = True,
##                                random_state = 1234)
##pca_rf.fit(train_comp,
##           train['label'])
##
### Generate predictions for 'train' using 'pca_rf'.
##train['pca_pred'] = pca_rf.predict(train_comp)
##
### Check training set accuracy for 'pca_rf'.
### Training set accuracy is 99.9976%.
### 'pca_pred' is extremely similar to 'baseline_pred'.
### Thus, the training error for 'pca_rf' is extremely similar to the training error for 'baseline_rf'.
##100 * sum(train['label'] == train['pca_pred']) / len(train['label'])
##
### Check cross-validation accuracy for 'pca_rf'.
### OOB accuracy is 90.8%.
### The OOB accuracy for 'pca_rf' is lower than the OOB accuracy for 'baseline_rf'.
##pca_rf.oob_score_
##
### Generate predictions for 'test' using 'pca_rf'.
##test['pca_pred'] = pca_rf.predict(test_comp)
##
### Test set accuracy cannot be checked as I do not have the answers for the test set.
##
