from enum import Enum

class FeatureType(Enum):
    TWO_VERTICAL=(1, 2)
    TWO_HORIZONTAL=(2, 1)
    THREE_HORIZONTAL=(3, 1)
    THREE_VERTICAL=(1, 3)
    FOUR=(2, 2)