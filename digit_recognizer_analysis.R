# This script uses pixel data to predict handwritten digits.

# PREP AND DATA IMPORT ----------------------------------------------------
# This section loads libraries and data.

  # Load libraries.
  library(readr)
  library(data.table)
  library(randomForest)
  library(dplyr)

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
  
  # Save 'train$label'.
  # This is necessary because once I turn it into a factor, it's difficult to get it use it another data tables, 
  # which will be necessary later.
  train_labels = train$label
  
  # Turn 'train$label' into a factor.
  train$label = as.factor(train$label)
  
  # Define model.
  model = label ~ .
  
  # Set seed for reproducibility.
  set.seed(1234)
  
  # Create random forest.
  # Note this takes a little while.
  # This script is mostly instructional and takes a while to run so I use a very small number of trees.
  baseline_rf = randomForest(model, 
                             data = train, 
                             ntree = 5)
  
  # Generate predictions for 'train' and 'test'.
  train$baseline_pred = predict(baseline_rf, 
                                newdata = train, 
                                type = 'response')
  test$baseline_pred = predict(baseline_rf, 
                               newdata = test, 
                               type = 'response')
  
  # Check training set accuracy (99.326%)
  prop.table(table(train$label == train$baseline_pred))
  
  # Check cross-validation accuracy.
  # For random forest, OOB score can be used for cross-validation.
  # OOB estimate of accuracy: 82.46% (indicates over-fitting)
  prop.table(table(train$label == baseline_rf$predicted))
  
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
  
  # Create data tables with 'train_labels' attached to the PCA outputs.
  full_pca_train = data.table(cbind(train_labels, 
                                    full_pca$x))
  full_pca_train = rename(full_pca_train, 
                          label = train_labels)
  full_pca_train$label = as.factor(full_pca_train$label)
  lim_pca_train = data.table(cbind(train_labels, 
                                   lim_pca$x))
  lim_pca_train = rename(lim_pca_train, 
                          label = train_labels)
  lim_pca_train$label = as.factor(lim_pca_train$label)
  
  # Train random forests using the output of 'full_pca_train' and 'lim_pca_train'.
  
    # Set seed for reproducibility.
    set.seed(1234)
    
    # Train random forests.
    full_pca_rf = randomForest(model, 
                               data = full_pca_train, 
                               ntree = 5)
    lim_pca_rf = randomForest(model, 
                              data = lim_pca_train, 
                              ntree = 5)
  
  # Generate training predictions from 'full_pca_rf' and 'lim_pca_rf'.
  train$full_pca_pred = predict(full_pca_rf, 
                                newdata = full_pca_train, 
                                type = 'response')
  train$lim_pca_pred = predict(lim_pca_rf, 
                               newdata = lim_pca_train, 
                               type = 'response')
  
  # Generate test predictions.
  
    # Project 'test' onto 'full_pca' and 'lim_pca' components.
    full_test_comp = predict(full_pca, 
                             newdata = test[, 
                                            !'baseline_pred', 
                                            with = F])
    lim_test_comp = predict(lim_pca, 
                            newdata = test[, 
                                           !'baseline_pred', 
                                           with = F])
    
    # Generate test predictions.
    test$full_pca_pred = predict(full_pca_rf, 
                                 newdata = full_test_comp, 
                                 type = 'response')
    test$lim_pca_pred = predict(lim_pca_rf, 
                                newdata = lim_test_comp, 
                                type = 'response')
    
  # Check training set accuracy for 'full_pca_pred' and 'lim_pca_pred'.
  # Training set accuracy is 97.88% and 98.9% for 'full_pca_pred' and 'lim_pca_pred' respectively.
  # This is less than the baseline model in both cases.
  prop.table(table(train$label == train$full_pca_pred))
  prop.table(table(train$label == train$lim_pca_pred))
  
  # Check OOB accuracy for 'full_pca_pred' and 'lim_pca_pred'.
  # OOB accuracy is 53.54% and 72.41% for 'full_pca_rf' and 'lim_pca_rf' respectively.
  # These are both worse than the basline model.
  prop.table(table(train$label == full_pca_rf$predicted))
  prop.table(table(train$label == lim_pca_rf$predicted))
      
  # Test set accuracy cannot be checked as I do not have the answers for the test set.