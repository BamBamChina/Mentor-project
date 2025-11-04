import numpy as np

class FromTwoToOne:
    def __init__(self, array_2d):
        self.array_2d = array_2d
    def transformation(self):
        self.array_2d = np.array(self.array_2d)
        self.array_1d = self.array_2d.flatten()
        return self.array_1d


x = FromTwoToOne([[1, 2, 3, 4], [5, 6, 7, 8]])
print(x.transformation())
