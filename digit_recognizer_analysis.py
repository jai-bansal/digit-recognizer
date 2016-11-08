# This script uses pixel data to predict handwritten digits.

#################
# PREP AND IMPORT
#################
# This section loads modules and data.

# Import modules.
import pandas as pd

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
