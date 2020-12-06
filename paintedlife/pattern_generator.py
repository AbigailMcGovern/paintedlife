import numpy as np

class PatternGenerator():

    def __init__(self, shape):
        """
        This class has a series of methods that give indices for setting
        GOL patterns on the grid. The majority of these will take a seed index
        which will be specified by a point on the grid (so that this can be
        chosen using napari points layer)
        """
        self.shape = shape
        self.viewer = None # will be napari viewer
        self.bounding_boxes = None # will be napari shapes layer




    def set_pattern(self, index, pattern):
        """
        Patterns will be specified by name. 
        Hopefully: add dropdown widget to choose the pattern that
        will be added at any point currently on the grid in points layer.
        -> generate bounding boxes (shapes layer) if the viewer is not none
        """
        pass


    def corner_squares(self):
        """
        Indicies for squares in the corners of the grid. 
        These will serve as markers for the corners, thus allowing you to
        paint within the grid.
        """
        shape = self.shape #fix this --- apparently too tired
        mesh = np.array(np.meshgrid([0, 1, shape[0] - 2, shape[0] - 1], 
                                    [0, 1, shape[1] - 2, shape[1] - 1]))
        combinations = mesh.T.reshape(-1, 2)
        y = combinations[:, 0]
        x = combinations[:, 1]
        return (y, x)


# add some common patterns