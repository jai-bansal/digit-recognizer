# This script uses pixel data to predict handwritten digits.

# PREP AND DATA IMPORT ----------------------------------------------------
# This section loads libraries and data.

  # Load libraries.
  library(readr)
  library(data.table)

  # Import training and test data.
  # Working directory must be set to the 'digit_recognizer' repository folder.
  train = data.table(read_csv('train.csv'))
  test = data.table(read_csv('test.csv'))

# DATA INSPECTION ---------------------------------------------------------
# This section inspects the data.
# It is commented out but results are listed.
  
  # View dimensions of data.
  # dim(train)
  # dim(test)
  
  # View summary of data.
  # This isn't very helpful because of the large number of columns.
  # summary(train)
  # summary(test)

  # Check 'train' and 'test' for NA values.
  # This takes a little while.
  # There are no NA values.
  # table(is.na(train))
  # table(is.na(test))
  
  # Check 'train' and 'test' for blank values.
  # This takes a little while.
  # There are no blank values.
  # table(train == '')
  # table(test == '')
  
  # Check the minimum and maximum values of 'train' and 'test'.
  # For both 'train' and 'test', the minimum and maximum values are 0 and 255 respectively.
  # So there are no negative values.
  # min(train)
  # max(train)
  # min(test)
  # max(test)
  
# BASELINE MODEL ----------------------------------------------------------
# This section implements a baseline random forest model with no feature engineering or modification.

  