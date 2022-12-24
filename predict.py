from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


d = {0 : 'airplane', 1 : 'automobile', 2: 'bird', 3 : 'cat', 4 : 'deer', 5 : 'dog', 6 : 'frog',
    7: 'horse', 8 : 'ship', 9 : 'truck'}


model = load_model('model/')


i_1 = cv2.imread('test/horse/0172.png')
i_2 = cv2.imread('test/automobile/0894.png')
i_3 = cv2.imread('test/frog/0203.png')

images = []
images.append(i_1)
images.append(i_2)
images.append(i_3)

for i in images:
    
    plt.imshow(i)
    plt.show()
    
    img_array = []

    # resize image
    resized = cv2.resize(i, (32, 32))


    test_image = np.ndarray(shape=resized.shape, dtype=resized.dtype, buffer=resized)

    img_array.append(test_image)
    x = np.asarray(img_array)

    result = model.predict(x)

    print('prediction: ', d[list(result[0]).index(max(list(result[0])))])
    

