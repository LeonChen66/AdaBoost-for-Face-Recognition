# AdaBoost for Face Recognition - Viola-Jones face detection
The Viola-Jones face detection scheme is one of the most well-known applications of the AdaBoost algorithm.

## Pipeline
Install the required python libraries for reading images, output visualization and scientific computing.
```
pip install -r requirements.txt
```
Run the main.py script to train the AdaBoost model, visualize the top ten selected features, draw the ROC curve, show the statistical results and the verification of performance improvement. 
```
python3 main.py
```

## File Description
* adaboost.py: Class for adaboost training
* haar.py: Class for extracting Haar feature
* feature_type.py: Enum class to represent the different types of Haar feature
* utils.py: Utils including data visualization, summed-area table calculation, adaboost counting vote.  