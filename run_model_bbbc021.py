import CNN_bbbc021
from keras import backend as K

"""
Below are the settings for our ResNet50;
Change model_type to "Inception_v3", set epochs to 7 and dims to (299,299,3) to run "our" InceptionV3;
Change model_type to "Inception_Resnet_v2", set epochs to 5, initial_lr to 0.001 and dims to (299,299,3) to run "our" InceptionResnetV3.

Note that the models may vary significantly from session to session due to stochastic processes of mini-batch gradient descent.

epochs_drop_lr is set to a greater value than epochs. Hence the initial_lr will be the learning rate throughout the entire training session.
"""


# Compounds
compounds = ["PP-2",           "AZ-J",                 "AZ-U",                                     # Epithelial
             "colchicine",     "vincristine",          "demecolcine",   "nocodazole",              # Microtubule destabilizers
             "docetaxel",      "taxol",                "epothilone B",                             # Microtubule stabilizers
             "ALLN",           "lactacystin",          "MG-132",        "proteasome inhibitor I",  # Protein degradation
             "anisomycin",     "emetine",              "cyclohexamide",                            # Protein synthesis
             "alsterpaullone", "bryostatin",           "PD-169316",                                # Kinase inhibitors
             "AZ138",          "AZ-C",                                                             # Eg-5 inhibitors
             "floxuridine",    "mitoxantrone",         "methotrexate",  "camptothecin",            # DNA-replication
             "etoposide",      "chlorambucil",         "cisplatin",     "mitomycin C",             # DNA-damage
             "simvastatin",    "mevinolin/lovastatin",                                             # Cholesterol-lowering
             "AZ841",          "AZ-A",                 "AZ258",                                    # Aurora kinase inhibitors
             "cytochalasin B", "latrunculin B",        "cytochalasin D"]                           # Actin disruptors

model_type = "ResNet50"
for compound in compounds:
    K.clear_session()
    clf = CNN_bbbc021.CNN_Model(cnn_model = model_type,
                                dims  = (224,224,3),
                                classes = 12,
                                regularization=0.0,
                                epochs = 5,
                                batch_size = 32,
                                initial_lr=0.0005,
                                drop_lr=0.5,
                                epochs_drop_lr=10.0,
                                momentum = 0.9,
                                weights  = "imagenet",
                                compound = compound,
                                gpus = 0)

    model = clf.compile_model()
    trained_model = clf.fit_model(model)
    clf.predict_model(trained_model)
