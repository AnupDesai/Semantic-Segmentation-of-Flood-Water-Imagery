# Semantic-Segmentation-of-Flood-Water-Imagery

Aim:

• The aim of this project is to design a Machine Learning model using PyTorch and PyTorch Lightning to detect the presence of flood water from the synthetic aperture radar (Sentinel-1) images.

<img width="423" alt="seg_1" src="https://user-images.githubusercontent.com/68967101/212528357-885291c8-ae2f-4367-8137-375e42f9306f.png">


<img width="422" alt="seg_2" src="https://user-images.githubusercontent.com/68967101/212528371-608df327-94a4-4987-a554-bb8c354d41ed.png">


<img width="420" alt="seg_3" src="https://user-images.githubusercontent.com/68967101/212528372-cec89f29-6f46-4e29-993a-a921ee678613.png">


Data:
The dataset includes 542 chips from flood events all across the world, each with two polarization bands. To get a better intuition about the dataset, it is better to visualize the number of images per location 

<img width="534" alt="seg_4" src="https://user-images.githubusercontent.com/68967101/212529009-09da659d-4349-495b-90a0-ba77974ad6b9.png">


Model:

• Implemented ResUnet++, a new state-of-the-art architecture for image segmentation with an “IoU” of 62.55%. The ResUnet++Model.py provides the entire model that is used to fit the dataset.

• The hyperparameters are tuned to fit the dataset to produce the optimum results and albumentation library is used to augment the data to avoid ovefitting.


