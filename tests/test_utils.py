import unittest
from face_rec.utils import *
from PIL import Image
import numpy as np


class IntegralImageTest(unittest.TestCase):

    def setUp(self):
        self.orig_img = np.array(Image.open('data/trainset/faces/face00001.png'), dtype=np.float64)
        self.int_img = to_integral_image(self.orig_img)

    def test_integral_calculation(self):
        assert self.int_img[1, 1] == self.orig_img[0, 0]
        assert self.int_img[-1, 1] == np.sum(self.orig_img[:, 0])
        assert self.int_img[1, -1] == np.sum(self.orig_img[0, :])
        assert self.int_img[-1, -1] == np.sum(self.orig_img)
        
    def test_area_sum(self):
        assert sum_region(self.int_img, (0, 0), (1, 1)) == self.orig_img[0, 0]
        assert sum_region(self.int_img, (0, 0), (-1, -1)) == np.sum(self.orig_img)

if __name__ == "__main__":
    unittest.main()