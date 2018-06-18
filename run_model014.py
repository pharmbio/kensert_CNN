import CNN_bbbc014

model_name = "ResNet50"

clf = CNN_bbbc014.CNN_Model(cnn_model = model_name,
                            dims  = (256,256,3),
                            regularization=0.0,
                            epochs = 1,
                            batch_size = 16,
                            lr=0.001,
                            momentum = 0.9,
                            weights = "imagenet")

clf.extend_model()
clf.eval()
