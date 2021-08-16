# Person-Reid-Triplet-Loss-Project
Person re-identification is the task of associating images of the same person taken from different cameras or from the same camera in different occasions, the process of person re-identification is shown in figure below, taken from this [paper survey](https://arxiv.org/abs/2001.04193).
<p align="center">
<img src="https://github.com/farhantandia/Person-Reid-Triplet-Loss-Project/blob/main/method%20of%20general%20reid.png"><br>
</p>
This repository contains the code of implementation person re-identification using triplet loss and SVM classifier with our custom dataset. The dataset which we are olympic swimmer dataset from YouTube in the top view angle, you can use your own dataset as well.

## Dependencies
- Tensorflow 2.3
- tensorflow-addons
- numpy
- imbalanced-learn (incase the data is imbalanced)
- scikit-learn 0.23.1

## Method
<p align="center">
<img src="https://github.com/farhantandia/Person-Reid-Triplet-Loss-Project/blob/main/method.png", width="700"><br>
</p>
From the detection model, we got a swimmer bounding box and we crop each swimmer image based on the bounding box then use it for the input of our deep learning model. In this case, we designed a compact CNN feature extractor which consist of 6 stacks of CNN with ReLU as activation function, batch normalization and max-pooling as shown in figure 19, the total parameters of the extractor is 166,000. This CNN will act as a feature extractor to get the 128 unique vector embeddings for triplet loss to calculate and optimize the distance of each swimmer embeddings. After getting the features for each swimmer, then we used a clustering model such as support vector machine (SVM) to classify the swimmer embeddings and get the correct ID. 

## Results
### T-SNE visualization 
<p align="center">
<img src="https://github.com/farhantandia/Person-Reid-Triplet-Loss-Project/blob/main/tsne%20results.png", width="700"><br>
</p>

### Accuracy comparison

| Metric | Validation Loss | Validation Accuracy |
| :------------------ | :--------------: | :-----------------: |
| CNN (Cross entropy) | 0.7284 | 80% |
| CNN (Triplet Loss) + SVM | 0.0014 | 99% |

