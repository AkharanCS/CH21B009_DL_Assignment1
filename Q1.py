import keras
from keras.datasets import fashion_mnist
import wandb

wandb.init(project="Assignment1_Q1")

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Plotting one image from each class
one_image = {}
c = 0
for cl,img in zip(train_labels,train_images):
    if cl not in one_image:
        one_image[cl] = img
        c += 1
    if c == 10:
        break

for cl,img in one_image.items():
    images = wandb.Image(img, caption="class"+str(cl))
    wandb.log({"all classes": images})

wandb.finish()
    


    
