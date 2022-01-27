import cv2
import trchime as tce

pstr = cv2.imread('picture/mnist_model_test_answer0(1).png', cv2.IMREAD_GRAYSCALE)


pstr = cv2.resize(pstr, (28, 28))
pstr = 255 - pstr

pstr = pstr / 255  # 28, 28

model = tce.loadmodel('model/mnist_model.pkl')

ans = model.predict(pstr.reshape(-1, 784))
print(ans)
print(ans.argmax(axis = 1))




