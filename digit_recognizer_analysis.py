# This script uses pixel data to predict handwritten digits.

#################
# PREP AND IMPORT
#################

# Import modules.
import pandas as pd

# Import training and test data.
# Working directory must be set to the 'digit_recognizer' repository folder.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
  
#################
# DATA INSPECTION
#################
