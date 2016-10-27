#### Summary:
This project attempts digit recognition using pixel information about handwritten digits using a data set originally from MNIST (and actually downloaded from kaggle.com). The goal is to use the pixel information to predict what number has been drawn. Multiple models are evaluated in R and Python.

#### Motivation:
I created this project to learn about object recognition.

### Contents:
The R and Python analyses are located in the "R" and "Python" branches respectively.

The "R" branch contains:
- the training and test data from kaggle.com ("train.csv") and ("test.csv") respectively
***
TO BE FILLED IN
***

The "Python" branch contains
- the training and test data from kaggle.com ("train.csv") and ("test.csv") respectively
***
TO BE FILLED IN
***

#### Dataset Details:
The data used is pixel information about handwritten single digits from MNIST (Modified National Institute of Standards and Technology).
I obtained the data from kaggle.com (https://www.kaggle.com/c/digit-recognizer/data). The training and test data are contained in 2 files, "train.csv" and "test.csv" respectively.

Below is the data description from kaggle.com:
"The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero)."

"train.csv" contains 42000 rows and 785 columns.
"test.csv" contains 28000 rows and 784 columns (no target label).

#### License:
GNU General Public License
