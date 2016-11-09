# This script uses pixel data to predict handwritten digits.

#################
# PREP AND IMPORT
#################
# This section loads modules and data.

# Import modules.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import training and test data.
# Working directory must be set to the 'digit_recognizer' repository folder.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
  
#################
# DATA INSPECTION
#################
# This section inspects the data.
# It's commented out but results are listed.

# View data dimensions.
# train.shape
# test.shape

# Check out data description.
# This isn't very helpful due to the large number of columns.
# train.describe()
# test.describe()

# View the type of each column.
# All columns are 'int64'.
# train.dtypes.value_counts()
# test.dtypes.value_counts()

# Check 'train' and 'test' for null values.
# Since all columns are 'int64' there is no need to check for blank values.
# There are no null values.
# train.isnull().sum().sum()
# test.isnull().sum().sum()

# Check the minimum and maximum values of 'train' and 'test'.
# For both 'train' and 'test', the minimum and maximum values are 0 and 255 respectively.
# So there are no negative values.
#train.min().min()
#train.max().max()
#test.min().min()
#test.max().max()

################
# BASELINE MODEL
#################
# This section implements a baseline random forest model with no feature engineering or modification.

# Create random forest classifier.
# This script is mostly instructional so I use a very small number of trees.
baseline_rf = RandomForestClassifier(n_estimators = 25,
                                     oob_score = True,
                                     random_state = 1234)

# Fit 'baseline_rf' on training data.
baseline_rf.fit(train.drop('label',
                           axis = 1),
                train['label'])

# Generate predictions for training and test data.
train['baseline_pred'] = baseline_rf.predict(train.drop('label',
                                                        axis = 1))
test['baseline_pred'] = baseline_rf.predict(test)

# Compute training set accuracy.
100 * sum(train['label'] == train['baseline_pred']) / len(train['baseline_pred'])

# Check cross-validation accuracy.
# For random forest, OOB score can be used for cross-validation.
baseline_rf.oob_score_

# Test set accuracy cannot be checked as I do not have the answers for the test set.

###########
# PCA MODEL
###########
# This section conducts Principal Components Analysis (PCA) before using a random forest model for prediction.
