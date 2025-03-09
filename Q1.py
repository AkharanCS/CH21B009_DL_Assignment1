import keras
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Plotting one iamge from each class
one_image = {}
c = 0
for cl,img in zip(train_labels,train_images):
    if cl not in one_image:
        one_image[cl] = img
        c += 1
    if c == 10:
        break

for cl,img in one_image.items():
    plt.imshow(img, cmap='gray') 
    plt.title(f"class: {cl}")
    plt.axis('off')
    plt.show()


    
