## Semantic-Segmentation-of-Flood-Water-Imagery

#• The aim of this project is to design a Machine Learning model to detect the presence of flood water from the synthetic aperture radar (Sentinel-1) images.

<img width="423" alt="seg_1" src="https://user-images.githubusercontent.com/68967101/212528357-885291c8-ae2f-4367-8137-375e42f9306f.png">


<img width="422" alt="seg_2" src="https://user-images.githubusercontent.com/68967101/212528371-608df327-94a4-4987-a554-bb8c354d41ed.png">


<img width="420" alt="seg_3" src="https://user-images.githubusercontent.com/68967101/212528372-cec89f29-6f46-4e29-993a-a921ee678613.png">


# Model:

• Implemented ResUnet++, a new state-of-the-art architecture for image segmentation with an “IoU” of 62.55%. The ResUnet++Model.py provides the entire model that is used to fit the dataset.

• The hyperparameters are tuned to fit the dataset to produce the optimum results and using albumentation library is used to augment the data to avoid ovefitting.


