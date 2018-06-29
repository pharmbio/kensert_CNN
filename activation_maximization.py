from vis.visualization import get_num_filters
from vis.utils import utils
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
#import imageio


# Uncomment the model you want to use.

# Fine-tuned model
model = load_model('model.h5')

# Pre-trained model
#model = ResNet50(weights='imagenet')


###---------------------------------------------------------------------------------------------------


layer_names = ['activation_10', 'activation_15', 'activation_35', 'activation_45']
count = 1
for layer_name in layer_names:
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Select 50 filters for each layer; either first 50 or randomly selected.
    filters = np.array(range(50)) #np.random.permutation(get_num_filters(model.layers[layer_idx]))[:50]

    for i in filters:
        losses = [
            (ActivationMaximization(layer_dict[layer_name], i), 1),
            (LPNorm(model.input), 6),
            (TotalVariation(model.input), 2)
        ]

        opt = Optimizer(model.input, losses)
        a,b,c = opt.minimize(max_iter=200, verbose=False, input_modifiers=[Jitter(0.05)])
        print(str(count) + '/200 DONE')
        count += 1
        a = Image.fromarray(a.astype("uint8"))
        a.save('act_max/' + layer_name + '_finetuned_' + str(i) + '.png')
