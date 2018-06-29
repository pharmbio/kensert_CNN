import CNN_bbbc021
#from keras.models import load_model
"""
To run Inceptionv3 or InceptionResnetv2, change model_type to Inception_v3 or Inception_Resnet_v2 respectively.

Number of epochs should be increased if more learning is needed.
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
    #trained_model.save('model.h5')
    clf.predict_model(trained_model)
