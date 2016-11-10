# This script uses pixel data to predict handwritten digits.

# PREP AND DATA IMPORT ----------------------------------------------------
# This section loads libraries and data.

  # Load libraries.
  library(readr)
  library(data.table)
  library(randomForest)

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
  
  # Turn 'train$label' into a factor.
  train$label = as.factor(train$label)
  
  # Define model.
  baseline_model = label ~ .
  
  # Set seed for reproducibility.
  set.seed(1234)
  
  # Create random forest.
  # Note this takes a little while.
  # This script is mostly instructional and takes a while to run so I use a very small number of trees.
  baseline_rf = randomForest(baseline_model, 
                             data = train, 
                             ntree = 5, 
                             importance = T)
  
  # Generate predictions for 'train' and 'test'.
  train$baseline_pred = predict(baseline_rf, 
                                newdata = train, 
                                type = 'response')
  test$baseline_pred = predict(baseline_rf, 
                               newdata = test, 
                               type = 'response')
  
  # Check training set accuracy.
  table(train$label == train$baseline_pred)
  prop.table(table(train$label == train$baseline_pred))
  
  # Check cross-validation accuracy.
  # For random forest, OOB score can be used for cross-validation.
  # OOB score is contained in the following output.
  baseline_rf
  
  # Test set accuracy cannot be checked as I do not have the answers for the test set.

# PCA MODEL ---------------------------------------------------------------
# This section conducts Principal Components Analysis (PCA) before using a random forest model for prediction.
  
  # Conduct principal component analysis on 'train'.
  # I conduct one PCA keeping all components and another removing some of the least useful componenets.
  # Normally, I would scale 'train' within the 'prcomp' command. But I do not for 2 reasons:
    # 1. All of the data is pixel data ranging from 0 to 255 so the scales are identical for all variables.
    # 2. Some of the columns are all a single number (I assume 0) which makes scaling fail.
  # This takes a little while.
  full_pca = prcomp(train[, 
                          !c('label', 'baseline_pred'),
                          with = F], 
                    center = T)
  lim_pca = prcomp(train[, 
                         !c('label', 'baseline_pred'), 
                         with = F], 
                   tol = 0.1, 
                   center = T)
  
  # View the summaries.
  summary(full_pca)
  summary(lim_pca)
  
  # Plot the variances associated with the principal components.
  plot(full_pca)
  plot(lim_pca)
  
  # Train random forests using the output of 'full_pca' and 'lim_pca'.
  full_pca_rf = 
  lim_pca_rf = 
    
  # Generate test predictions.
  
    # Apply 'full_pca' and 'lim_pca' to 'test'.
    full_test_comp = predict(full_pca, 
                             newdata = test[, 
                                            !'baseline_pred', 
                                            with = F])
    lim_test_comp = predict(lim_pca, 
                            newdata = test[, 
                                           !'baseline_pred', 
                                           with = F])
    
    # Generate test predictions...
    test$full_pca_pred = 
    test$lim_pca_pred =
      
  # Test set accuracy cannot be checked as I do not have the answers for the test set.

  