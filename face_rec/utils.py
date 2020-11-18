import numpy as np
import cv2
from face_rec.feature_type import FeatureType
from functools import partial
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

"""
For to_integral_image and sum_region, please refer to Summed-area table
https://en.wikipedia.org/wiki/Summed-area_table
"""

def to_integral_image(img):
    row_sum = np.zeros(img.shape)
    integral_image = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            row_sum[i, j] = row_sum[i-1, j] + img[i, j]
            integral_image[i+1, j+1] = integral_image[i+1, j-1+1] + row_sum[i, j]
    return integral_image


def sum_region(integral_img, top_left, bottom_right):
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return integral_img[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    return integral_img[bottom_right] - integral_img[top_right] - integral_img[bottom_left] + integral_img[top_left]


def load_images(path):
    image_list = []
    for file_name in os.listdir(path):
        if file_name.endswith('.png'):
            img = np.array(cv2.imread(os.path.join(path, file_name), 0), dtype=np.float64)
            img /= img.max()
            image_list.append(img)
            
    return image_list


def vis_haar(classifiers, base_img):
    img_list = []
    for classifier in classifiers:
        img = np.copy(base_img)
        img -= img.min()
        img /= img.max()
        img *= 255
        if classifier.type == FeatureType.TWO_VERTICAL:
            for x in range(classifier.width):
                for y in range(classifier.height):
                    if y >= classifier.height/2:
                        sign = 255
                    else:
                        sign = 0
                    img[classifier.top_left[1] + x, classifier.top_left[0] + y] = sign
        elif classifier.type == FeatureType.TWO_HORIZONTAL:
            for x in range(classifier.width):
                if x >= classifier.width/2:
                    sign = 255
                else:
                    sign = 0
                for y in range(classifier.height):
                    img[classifier.top_left[0] + x, classifier.top_left[1] + y] = sign
        elif classifier.type == FeatureType.THREE_HORIZONTAL:
            for x in range(classifier.width):
                if x >= classifier.width/3 and x < classifier.width*2/3:
                    sign = 255
                else:
                    sign = 0
                for y in range(classifier.height):
                    img[classifier.top_left[0] + x, classifier.top_left[1] + y] = sign
        elif classifier.type == FeatureType.THREE_VERTICAL:
            for x in range(classifier.width):
                for y in range(classifier.height):
                    if y >= classifier.height/3 and y<classifier.height*2/3:
                        sign = 255
                    else:
                        sign = 0
                    img[classifier.top_left[0] + x, classifier.top_left[1] + y] = sign 
        elif classifier.type == FeatureType.FOUR:
            for x in range(classifier.width):
                for y in range(classifier.height):
                    if (y >= classifier.height/2 and x >= classifier.width/2) or (y < classifier.height/2 and x < classifier.width/2):
                        sign = 255
                    else:
                        sign = 0
                    img[classifier.top_left[0] + x, classifier.top_left[1] + y] = sign

        img_list.append(img)      
    
    return img_list


def ensemble_vote(int_img, classifiers):
    if sum(classifier.get_vote(int_img) for classifier in classifiers) >= 0:
        return 1
    else:
        return 0


def ensemble_vote_all(imgs, classifiers):
    vote_partial = partial(ensemble_vote, classifiers=classifiers)
    return list(map(vote_partial, imgs))


def ensemble_score(int_img, classifiers):
    return sum(classifier.get_vote(int_img) for classifier in classifiers)


def ensemble_score_all(imgs, classifiers):
    vote_partial = partial(ensemble_score, classifiers=classifiers)
    return list(map(vote_partial, imgs))


def plot_confusion_matrix(correct_faces, incorrect_faces, correct_non_faces, incorrect_non_faces):
    array = [[correct_faces, incorrect_faces], [incorrect_non_faces, correct_non_faces]]
    df_cm = pd.DataFrame(array, index = ["Face", "Non-face"],
                  columns = ["Face", "Non-face"])
    
    plt.figure(figsize = (10,7))
    return sn.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu")
    
