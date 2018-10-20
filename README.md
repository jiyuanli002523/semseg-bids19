2D Semantic Segmentation
==============================

2D semantic segmentation (Vaihingen dataset)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Placeholder to put training and validation data
    │
    ├── docker             <- Directory for Dockerfile(s)
    │
    ├── models             <- Placeholder for trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── 2dsemseg           <- Source code for use in this project.
        ├── __init__.py    <- Makes 2dsemseg a Python module
        │
        ├── augmentation.py  <- to apply augmentation on original data
        │
        ├── data_io.py       <- to generate the Training and Validation Set from original data
        │
        ├── evaluate_network.py    <- to test trained netowrk
        │
        └── model_generator.py     <- model generator :-)
        │
        └── resnet_edit.py         <- ResNet50 for Keras
        │
        └── storeincsv.py          <- module to write csv summary
        │
        └── train_resnet50_fcn.py  <- main code for training


