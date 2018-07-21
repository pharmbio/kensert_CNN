import CNN_bbbc014

"""
Change model_type to "Inception_v3" or "Inception_Resnet_v2" to run InceptionV3 or InceptionResnetV2 respectively.

Note that the models may vary significantly from session to session due to stochastic processes of mini-batch gradient descent.
"""


model_type = "ResNet50"

clf = CNN_bbbc014.CNN_Model(cnn_model = model_type,
                            dims  = (256,256,3),
                            regularization=0.0,
                            epochs = 1,
                            batch_size = 8,
                            lr=0.001,
                            momentum = 0.9,
                            weights = "imagenet")

clf.extend_model()
clf.fit_and_eval()
