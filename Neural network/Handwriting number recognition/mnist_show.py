import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# flatten = true: 1차원 배열로 저장
# normalize = false: 입력 이미지 픽셀 값을 0.0 ~1.0까지의 정규화 하지 않고 입력 이미지 픽셀 값 그대로 유지

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # reshape(): 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)