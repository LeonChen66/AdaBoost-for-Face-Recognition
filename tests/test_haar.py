import unittest
from face_rec.haar import Haar
from face_rec.feature_type import FeatureType
from face_rec.utils import *
import numpy as np
from PIL import Image


class HaarTest(unittest.TestCase):

    def setUp(self):
        img_arr = np.array(Image.open('data/trainset/faces/face00001.png'), dtype=np.float64)
        self.int_img = to_integral_image(img_arr)

    def test_two_vertical(self):
        feature = Haar(FeatureType.TWO_VERTICAL, (0, 0), 19, 19, 100000, 1)
        left_area = sum_region(self.int_img, (0, 0), (19, 12))
        right_area = sum_region(self.int_img, (0, 12), (19, 19))
        expected = 1 if feature.threshold * feature.polarity > left_area - right_area else 0
        assert feature.get_vote(self.int_img) == expected
        
    def test_two_vertical_fail(self):
        feature = Haar(FeatureType.TWO_VERTICAL, (0, 0), 19, 19, 100000, 1)
        left_area = sum_region(self.int_img, (0, 0), (19, 12))
        right_area = sum_region(self.int_img, (0, 12), (19, 19))
        expected = 1 if feature.threshold * -1 > left_area - right_area else 0
        assert feature.get_vote(self.int_img) != expected

    def test_two_horizontal(self):
        feature = Haar(FeatureType.TWO_HORIZONTAL, (0,0), 19, 19, 100000, 1)
        left_area = sum_region(self.int_img, (0, 0), (19, 12))
        right_area = sum_region(self.int_img, (0, 12), (19, 19))
        expected = 1 if feature.threshold * feature.polarity > left_area - right_area else 0
        assert feature.get_vote(self.int_img) == expected

    def test_three_horizontal(self):
        feature = Haar(FeatureType.THREE_HORIZONTAL, (0, 0), 19, 19, 100000, 1)
        left_area = sum_region(self.int_img, (0, 0), (8, 19))
        middle_area = sum_region(self.int_img, (8, 0), (16, 19))
        right_area = sum_region(self.int_img, (16, 0), (19, 19))
        expected = 1 if feature.threshold * feature.polarity > left_area - middle_area + right_area else 0
        assert feature.get_vote(self.int_img) == expected

    def test_three_vertical(self):
        feature = Haar(FeatureType.THREE_VERTICAL, (0, 0), 19, 19, 100000, 1)
        left_area = sum_region(self.int_img, (0, 0), (19, 8))
        middle_area = sum_region(self.int_img, (0, 8), (19, 16))
        right_area = sum_region(self.int_img, (0, 16), (19, 19))
        expected = 1 if feature.threshold * feature.polarity > left_area - middle_area + right_area else 0
        assert feature.get_vote(self.int_img) == expected

    def test_four(self):
        feature = Haar(FeatureType.THREE_HORIZONTAL, (0, 0), 19, 19, 100000, 1)
        top_left_area = sum_region(self.int_img, (0, 0), (12, 12))
        top_right_area = sum_region(self.int_img, (12, 0), (19, 12))
        bottom_left_area = sum_region(self.int_img, (0, 12), (12, 19))
        bottom_right_area = sum_region(self.int_img, (12, 12), (19, 19))
        expected = 1 if feature.threshold * feature.polarity > top_left_area - top_right_area - bottom_left_area + bottom_right_area else 0
        assert feature.get_vote(self.int_img) == expected


if __name__ == "__main__":
    unittest.main()