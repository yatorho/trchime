import tensorflow as tf
from trchime.datasets import cifar_set
import trchime as tce

#print(train_images[0])
#print("Network Accuracy: " + str(test_acc))
# plt.imshow(train_images[0], cmap=plt.cm.binary) #greyscale
# plt.imshow(train_images[0]) #neon
# plt.show()

url = '../datasets/datasets/cifar_datasets.npz'

cifar = cifar_set(url)  # pass your dataset url

train_images, train_labels, test_images, test_labels = cifar.load()
train_images, test_images = train_images.transpose(0, 2, 3, 1), test_images.transpose(0, 2, 3, 1)

# train_labels, test_labels = train_labels.reshape((50000,)), test_labels.reshape((10000,))  # reshape y sets
# train_labels, test_labels = tce.one_hot(train_labels, 100).data, tce.one_hot(test_labels, 100).data  # one hot code y sets
# print(train_labels.shape)
train_images = train_images/255
test_images = test_images/255
classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1080, activation='relu'),
    tf.keras.layers.Dense(1080, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10)

model.summary()

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(test_acc)
# # print(test_images)
# prediction = model.predict(test_images)
# answer = np.argmax(prediction[0])
# print(classes[answer])
# # print(train_images[0])
# plt.imshow(train_images[0])
# plt.show()







