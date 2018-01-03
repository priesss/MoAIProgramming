import inspect, os #Imports
import numpy as np

debug = True #Global Switch for Debug-Info

#Determine working directory and gridpath
scriptpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
gridsubpath = "/Grids/3by4.grid"
gridpath = scriptpath + gridsubpath

#Debug-Info
if debug:
    print("Scriptpath: ")
    print(scriptpath.encode('utf-8'))
    print("Gridpath: ")
    print(gridpath.encode('utf-8'))


#Read Grid-File as 2-dimensional array 'grid' from 'gridpath'
with open(gridpath) as gridfile:
    grid = [line.split() for line in gridfile]


#Debug-Info
if debug:
    print("Raw-Grid: ")
    print(grid)
    print("Grid cleaned-up: ")
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end=' ')
        print()

