from itertools import permutations
from functools import partial 
import napari
from napari import Viewer
import numpy as np
from magicgui import magicgui
from .pattern_generator  import PatternGenerator # nothing lives here yet
from scipy import sparse
from time import time



class GameOfLife(PatternGenerator):

    def __init__(self, shape, seed=12, timeit=True, verbose=False):
        """

        Parameters
        ----------
        shape: tuple of int
            Shape of the Game of Life world (y, x)
        seed: int
            Seed for the labels layer in napari.
            Determines the colour of the labels. 


        Attributes
        ----------
        _grid: scipy.sparse.lil_matrix
            current grid, possibly not up to date with painted
            cells from viewer 
            NOTE: LIL choosen as facilitates fancy indexing
        grid: lil_matrix (property)
            current grid corresponding to the final frame displayed
            in the viewer
        _history: list of lil_matrix
            Game history, possibly not up to date with painted
            cells from viewer
        history: np.array
            Current array displayed in the labels layer. Perhaps
            for very large/long games dask from delayed would be 
            better. Unfortunately, this doesnt support fancy indexing
            so can't be used if you want to paint cells. 
        viewer: None or napari.Viewer
            When the class is called, the Viewer is instantiated
        seed: int
            Seed for napari labels layer


        TODO: construct grid of some description 
              (difficult to paint bigger patterns at the moment)

        TODO: initialise with existing or saved (npz) matrix

        TODO: pattern adding tools (see pattern_generator.py)

        TODO: copy paste patterns (labels layer)

        TODO: autogeneration of patterns for patterns lib from painted patterns
              would want to probably paint an area to convert and use live cells 
              to find indices to generate code for the pattern
        """
        super().__init__(shape)
        grid = np.zeros(shape, dtype=bool)
        self._grid = sparse.lil_matrix(grid, dtype=bool)
        self._history = [self._grid.copy(), ]
        self.viewer = None 
        self.seed = seed
        self.timeit = timeit
        self.verbose = verbose


    @property
    def grid(self):
        if self.viewer is None:
            return self._grid
        else:
            self.viewer.layers['Game of Life'].refresh()
            grid = sparse.lil_matrix(
                                            self.viewer.layers['Game of Life'].
                                            data[-1].
                                            astype(bool)
                                            )
            return grid


    @property
    def history_shape(self):
        t = len(self._history)
        y = self.shape[0]
        x = self.shape[1]
        return (t, y, x)


    @property
    def history(self):
        array = self._construct_array()
        return array


    def _construct_array(self):
        array = np.concatenate([[g.toarray()] for g in self._history])
        return array


    def __call__(self, show_corners=False, backlit=True):
        """
        """
        if show_corners:
            self._grid[self.corner_squares()] = True
            self._update()
        with napari.gui_qt():
            self.viewer = napari.Viewer()
            if backlit:
                square = [[0, 0], 
                          [0, self.shape[1]], 
                          [self.shape[0], self.shape[1]],
                          [self.shape[0], 0]]
                self.viewer.add_shapes(square, name='backlight', opacity=0.05)
            self._add_GOL_to_viewer()
            self.viewer.bind_key('u', self._update)
            self._evolve_gui()

    
    def _evolve_gui(self):
        e = partial(evolve, self)
        e = partial(e, self.viewer)
        e = partial(e, self.timeit)
        e = magicgui(e, call_button='Evolve', layout='form')
        evolve_gui = e.Gui()
        self.viewer.window.add_dock_widget(evolve_gui)
        self.viewer.layers.events.changed.connect(lambda x: evolve_gui.refresh_choices())


    def _update(self, viewer=None, in_evolve=False):
        """
        Update the grid for a single generation
        """
        t = time()
        if viewer is None:
            viewer = self.viewer
        elif not in_evolve:
            self.update_from_labels_layer()
        changable = self._find_changable(in_evolve)
        idx = sparse.find(changable == True)
        next_state = []
        for i in range(len(idx[0])):
            y, x = idx[0][i], idx[1][i]
            s = self._next_state(y, x)
            next_state.append(s)
        self._grid[idx[0], idx[1]] = next_state
        if not in_evolve and self.timeit and self.verbose:
            print(f'uptate state sparse: {time()-t} s')
        self._history.append(self._grid.copy())
        if not in_evolve:
            self.update_viewer()


    def _find_changable(self, in_evolve):
        """
        Find the alive cells and their neighbors. 
        This forms a mask for the update computation
        """
        #indicies = np.where(self._grid == True)
        t = time()
        indicies = sparse.find(self._grid == 1)
        y = np.concatenate([indicies[0], 
                            indicies[0] + 1, 
                            indicies[0], 
                            indicies[0] - 1, 
                            indicies[0] + 1, 
                            indicies[0] - 1,
                            indicies[0] + 1, 
                            indicies[0] - 1
                            ])
        x = np.concatenate([indicies[1] + 1, 
                            indicies[1], 
                            indicies[1] - 1, 
                            indicies[1],
                            indicies[1] -1, 
                            indicies[1] + 1,
                            indicies[1] + 1, 
                            indicies[1] - 1
                            ])
        idx = self._get_idx(y, x)
        changable = self._grid.copy()
        changable[idx] = True
        if not in_evolve and self.timeit and self.verbose:
            print(f"find changable: {time() - t} s")
        return changable


    def _next_state(self, y, x):
        new_y = np.array([y - 1, y, y + 1, y, y + 1, y - 1, y + 1, y - 1])
        new_x = np.array([x, x - 1, x, x + 1, x - 1, x + 1, x + 1, x - 1])
        ind = self._get_idx(new_y, new_x)
        neighbors = [i for i in self._grid[ind].data]
        n_neighbors = np.sum(neighbors)
        # current state
        state = self._grid[y, x]
        if state == 1: # alive cell
            if n_neighbors >= 4 or n_neighbors <= 1:
                s = 0 # overcrowding or isolation
            else:
                s = 1 # 2 or 3 -> remain alive
        else: # dead cell
            if n_neighbors == 3:
                s = 1 # 3 -> birth
            else:
                s = 0 # otherwise remain dead
        return s


    def _get_idx(self, y, x):
        """
        Remove indicies outside of the grid
        """
        y_max = self._grid.shape[0]
        x_max = self._grid.shape[1]
        # remove anything that falls outside of the grid
        keep_idx = np.concatenate([np.where((y >= 0) & (y < y_max))[0], 
                               np.where((x >= 0) & (x < x_max))[0]])
        # only keep y,x pairs for which both values are within grid
        unique, counts = np.unique(keep_idx, return_counts=True)
        keep_idx = unique[np.where(counts >= 2)]
        y = y[keep_idx]
        x = x[keep_idx]
        return (y, x)


    def update_viewer(self):
        viewer = self.viewer
        t = time()
        if viewer is not None:
            self._add_GOL_to_viewer()
            step = len(self._history) - 1
            step = int(step)
            viewer.dims.set_current_step(0, step)
            if self.timeit and self.verbose:
                print(f'uptate viewer: {time()-t} s')


    def _add_GOL_to_viewer(self):
        try:
            del self.viewer.layers['Game of Life']
        except ValueError:
            pass
        self.viewer.add_labels(self.history, name='Game of Life', opacity=1.0)
        self.viewer.layers["Game of Life"].seed = self.seed
        self.viewer.layers["Game of Life"].brush_shape = 'SQUARE'
        self.viewer.layers["Game of Life"].brush_size = 1

    
    def update_from_labels_layer(self):
        """
        Compare grid (property: last frame of history from napari) and
        _grid (describes current status). If different set grid equal to
        the displayed (painted) data and add the new grid to _history. 
        """
        diff = self.grid != self._grid
        if diff.max():
            self._grid = self.grid
            self._history.append(self._grid.copy()) 



def evolve(gol: GameOfLife, viewer : Viewer,  timeit: bool, n_generations: int):
    t = time()
    gol.update_from_labels_layer()
    for _ in range(n_generations):
        gol._update(viewer, in_evolve=True)
    gol.update_viewer()
    if timeit:
        print(f'Evolve for {n_generations} generations: {time() - t} s')