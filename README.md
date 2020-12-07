# paintedlife
[![PyPI version shields.io](https://img.shields.io/pypi/v/paintedlife.svg)](https://pypi.org/project/paintedlife/0.0.1/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

Conway's Game of Life (`napari` style). Have you ever dreamed of painting patterns into the game of life? I know I have. 

## Installation
paintedlife is available via pip 
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
