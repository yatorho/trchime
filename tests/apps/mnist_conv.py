import trchime as tce
import cv2


def showlayer1(model_, pstr_, nums):
    c1 = tce.nn.Conv2D(pstr_.reshape((1, 1, 28, 28)), model_.filter1, model_.bias1)  # 1, 3, 26, 26
    c1.assign_mul(255)
    cv2.imshow('2', cv2.resize(c1.astype(tce.uint8).data[0, nums, :, :], (480, 640)))
    cv2.waitKey(0)

pstr = cv2.imread('../picture/mnist_model_test_answer9(6).png', cv2.IMREAD_GRAYSCALE)

# print(pstr.shape)
pstr = cv2.resize(pstr, (28, 28))
pstr = 255 - pstr

pstr = pstr / 255  # 28, 28

model = tce.loadmodel('../model/mnist_conv.pkl')  # load model trained previously

showlayer1(model, pstr, 1)

ans = model.predict(pstr.reshape((1, 1, 28, 28)))  # call model's predict fucntion
# print(ans)
print("predicted:", ans.argmax(axis = 1).data)








