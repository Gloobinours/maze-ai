from flask_restful import Resource
from flask import request, jsonify
import json
import sys
sys.path.insert(0, '../')
from Maze import Maze

class MazeEndpoint(Resource):

    def get(self, maze_size):
        maze: Maze = Maze(maze_size, 1, a_seed= 2)
        mtx = []
        for x in range(maze_size):
            mtx.append([])
            for y in range(maze_size):
                mtx[x].append(maze.grid[x][y].state.value)
        return jsonify(matrix=mtx)