# - Image classification using a Keras model 
# - The dataset used is the fashion MNIST dataset

# Installing python
path_to_python <- install_python()

# Importing the relevant libraries.
library(reticulate)
library(tensorflow)
library(keras)
library(tidyr)
library(ggplot2)
# Checking if the tensorflow package was successful y installed.
tf$constant("Hello Tensorflow!")


# Importing the dataset.
# - The dataset contains 70,000 observations.
fashion_mnist <- dataset_fashion_mnist()

# - Dividing the dataset into train and test 
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# checking the dimensions of the train and test sets images and labels
dim(train_images)
dim(train_labels)

dim(test_images)
dim(test_labels)



# Exploring the dataset

# Checking the 1st image
image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

# Checking the second image
image_2<- as.data.frame(train_images[2, ,])
colnames(image_2) <- seq_len(ncol(image_2))
image_2$y <- seq_len(nrow(image_2))
image_2 <- gather(image_2, "x", "value", -y)
image_2$x <- as.integer(image_2$x)

# Plotting image 1, the boot (The pixel falls between 0 and 255)
ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# plotting image 2, a T-shirt (The pixel falls between 0 and 255)
ggplot(image_2, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# As they are, these images cannot be fed into a model. 
# It is therefore vital that we scale the pixels to values between 0 and 1.

# Scalling the dataset.

train_images <- train_images / 255
test_images <- test_images / 255


# Declaring the class names: image names
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')


# plotting the 1st 25 images with class names
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')

for (i in 1:25) {
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}

# Modelling

# Building the Keras model.
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'tanh') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compiling the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)


# Fitting the model
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)

# Evaluating the model
score <- model %>% evaluate(test_images, test_labels, verbose = 0)

# Generating the model stats
cat('Test loss:', score["loss"], "\n")
cat('Test accuracy:', score["accuracy"], "\n")

# Carrying out the predictions.
predictions <- model %>% predict(test_images)
predictions[1, ]
which.max(predictions[1, ])

# plotting the 1st 25 predictions. 
test_labels[1]
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')

for (i in 1:25) {
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}


# Making a prediction using the test dataset.
img <- test_images[1, , , drop = FALSE]
dim(img)

predictions <- model %>% predict(img)
predictions

prediction <- predictions[1, ] - 1
which.max(prediction)
