# This script uses pixel data to predict handwritten digits using
# a convolutional neural network (CNN) using the 'tensorflow' package.

# This script borrows generously from the Udacity Deep Learning Course, Assignment 4.

# IMPORT LIBRARIES --------------------------------------------------------
# This section imports libraries.
library(readr)
library(data.table)
library(dplyr)
library(tensorflow)

# IMPORT DATA -------------------------------------------------------------
# This section imports data.
# Import training and test data.
# Working directory must be set to the 'digit_recognizer' repository folder.
train = data.table(read_csv('train.csv'))
test = data.table(read_csv('test.csv'))

# PREP DATA ---------------------------------------------------------------
# This section preps the data for model training.

  # Change 'train$label' to type 'factor'.
  train$label = as.factor(train$label)

  # Get training and test data (not labels).
  # There are no labels for the test data.
  train_data = as.matrix(select(train, 
                                -c(label)))
  test_data = as.matrix(test)
  
  # Images currently have 1 row per image.
  # For a CNN, I need square/rectangle images so the filter can sweep along.
  train_data = array(train_data, 
                     dim = c(nrow(train), 28L, 28L, 1L))
  test_data = array(test_data, 
                    dim = c(nrow(test), 28L, 28L, 1L))

  # Turn training labels into one-hot vectors.
  # Training labels are already integers.
  # There are no labels for the test data.
  train_labels_one_hot = model.matrix(~ label - 1, 
                                      data = train)
  
# SET MODEL PARAMETERS ----------------------------------------------------
# This section sets model parameters.
  
  # Specify number of classes (output labels).
  classes = 10L
  
  # Specify image size in pixels.
  image_size = 28L
  
  # Specify number of channels (1 since the images are greyscale).
  channels = 1L
  
  # Set # of iterations to run.
  steps = 1000
  
  # Set batch size.
  batch_size = 16
  
  # Set patch size for filter.
  patch_size = 5L
  
  # Set output depth for filters.
  output_depth = 16L
  
  # Set number of hidden nodes.
  hidden = 64L
    
# CREATE MODEL ------------------------------------------------------------
# This section creates a CNN using 'tensorflow'.
  
  # Set random seed.
  tf$set_random_seed(20170914)
  
  # Input placeholders for batch processes and full training data.
  train_place = tf$placeholder(tf$float32,
                               shape = c(batch_size, image_size, image_size, channels))
  train_labels = tf$placeholder(tf$float32,
                                shape = c(batch_size, classes))
  full_train_labels = tf$constant(train_labels_one_hot)
  full_train_data = tf$constant(train_data, 
                                dtype = tf$float32)
  full_test_data = tf$constant(test_data, 
                               dtype = tf$float32)
  
  # Specify convolution filters and biases.
  filter_1 = tf$Variable(tf$truncated_normal(c(patch_size,
                                              patch_size,
                                              channels,
                                              output_depth),
                                             stddev = 0.1))
  filter_1_bias = tf$Variable(tf$zeros(c(output_depth)))

  filter_2 = tf$Variable(tf$truncated_normal(c(patch_size,
                                               patch_size,
                                               output_depth,
                                               output_depth),
                                             stddev = 0.1))
  filter_2_bias = tf$Variable(tf$constant(rep(1, output_depth)))
  
  # Create variables for modified image dimensions.
  mod_imdim = as.integer(floor(image_size / 4))
  
  # Specify weights and biases for fully connected layers.
  w_1 = tf$Variable(tf$truncated_normal(c(mod_imdim * mod_imdim * output_depth,
                                          hidden),
                                        stddev = 0.1))
  b_1 = tf$Variable(tf$constant(rep(1, hidden)))

  w_2 = tf$Variable(tf$truncated_normal(c(hidden, classes),
                                        stddev = 0.1))
  b_2 = tf$Variable(tf$constant(rep(1, classes)))
  
  # Define training operations.
  # This is only done in a function because the code this script is based on had it this way :p
  training_computations = function(data)
    
  {
    
    # Specify first convolution.
    conv_1 = tf$nn$conv2d(data,
                          filter_1,
                          c(1, 2, 2, 1),
                          padding = 'SAME')
    
    # Compute 'relu' on 'conv_1'.
    relu_1 = tf$nn$relu(conv_1 + filter_1_bias)

    # Specify second convolution.
    conv_2 = tf$nn$conv2d(relu_1,
                          filter_2,
                          c(1, 2, 2, 1),
                          padding = 'SAME')

    # Compute 'relu' on 'conv_2'.
    relu_2 = tf$nn$relu(conv_2 + filter_2_bias)
    
    # Save dimensions of 'relu_2'.
    shape = relu_2$get_shape()$as_list()

    # Reshape 'relu_2' for fully connected layer operations.
    relu_2_reshape = tf$reshape(relu_2,
                                shape = c(shape[1], (shape[2] * shape[3] * shape[4])))

    # Conduct first fully connected layer matrix multiplication and 'relu' operation.
    full_1 = tf$nn$relu(tf$matmul(relu_2_reshape, w_1) + b_1)

    # Return second fully connected layer matrix multiplication.
    return(tf$matmul(full_1, w_2) + b_2)
    
  }
  
  # Execute training computation.
  logits = training_computations(train_place)
  
  # Specify loss.
  loss = tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(labels = train_labels,
                                                                logits = logits))
  
  # Specify optimizer.
  optimizer = tf$train$GradientDescentOptimizer(0.01)$minimize(loss)
  
  # Generate predictions for training and test set.
  train_pred = tf$nn$softmax(logits)
  full_train_pred = tf$nn$softmax(training_computations(full_train_data))
  test_pred = tf$nn$softmax(training_computations(full_test_data))
  
  # Launch the graph and initialize the variables.
  session = tf$Session()
  session$run(tf$global_variables_initializer())
  
  # Iterate.
  for (step in 1:steps)
    
  {
    
    # Select index to pick batch.
    # Just having the expression before the '%' sign makes intuitive sense.
    # But what if there were so many iterations that the training data ran out??
    # That's what the stuff after the '%' sign does. Why that specific form? No clue.
    cutoff = (step * batch_size) %% (dim(train_data)[1] - batch_size)
    
    # Generate batch.
    batch_data = train_data[(cutoff + 1) : (cutoff + batch_size), 1:28, 1:28, 1, 
                            drop = F]
    batch_labels = train_labels_one_hot[(cutoff + 1) : (cutoff + batch_size), 1:10]
    
    # Run optimizer (defined above) using 'feed_dict' to feed in the batch.
    session$run(optimizer, 
                feed_dict = dict(train_place = batch_data, 
                                 train_labels = batch_labels))
    
    # Print progress and metrics.
    if (step %% 100 == 0)
      
    {
      
      # Print metrics.
      print(paste0('Step ', step, ':'))
      print(paste0('Loss: ', 
                   round(session$run(loss, 
                                     feed_dict = dict(train_place = batch_data, 
                                                      train_labels = batch_labels)), 2)))
      print(paste0('Training Set Accuracy: ', 
                 round(session$run(tf$reduce_mean(tf$cast(tf$equal(tf$argmax(train_pred, 1L), 
                                                                   tf$argmax(batch_labels, 1L)), 
                                                          dtype = tf$float32)), 
                                   feed_dict = dict(train_place = batch_data, 
                                                    train_labels = batch_labels)), 2)))
      
    }
    
  }
  
  
  
