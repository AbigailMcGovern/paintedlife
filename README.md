# Conway's Game of Life

Conway's Game of Life (`napari` style). 

## Installation
paintedlife is available via pip (0.0.1)
```bash
pip install paintedlife
```

## Usage
```Python
from paintedlife import GameOfLife
shape = (100, 100)
gol = GameOfLife(shape)
gol()
# now just paint your preferred life
```

## Notes
- if in ipython you can access the sparse versions of the displayed arrays using `gol._history`
- save sparse array from history using `scipy.sparse.save_npz`

## To Do List
- Support initalisation with saved grid
- tools for adding known patterns
- pattern library
- grid lines?
- autogenerate code for pattern library from pattern found or painted on grid
