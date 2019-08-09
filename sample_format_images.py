import os
import skimage
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images,labels

ROOT_PATH = "data"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
images28 = [skimage.transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = skimage.color.rgb2gray(images28)

traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)
    img = images28[traffic_signs[i]]
    print("shape: {0}, min: {1}, max: {2}".format(img.shape,img.min(),img.max()))
plt.show()
