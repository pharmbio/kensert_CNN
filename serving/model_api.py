import tensorflow, numpy as np
from PIL import Image
from keras.models import load_model

model = load_model("saved_bbbc014_ResNet50.h5")

def predict(input):
    image = None
    with Image.open(input) as img_pointer:
    #image = Image.open(input)
        image = np.array(img_pointer)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    if image:
        return model.predict(image)

    else:
        print("Could not load image")
        return None