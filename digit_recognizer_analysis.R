# This script uses pixel data to predict handwritten digits.

# PREP AND DATA IMPORT ----------------------------------------------------

  # Load libraries.
  library(readr)
  library(data.table)

  # Import training and test data.
  # Working directory must be set to the 'digit_recognizer' repository folder.
  train = data.table(read_csv('train.csv'))
  test = data.table(read_csv('test.csv'))

# DATA INSPECTION ---------------------------------------------------------

  