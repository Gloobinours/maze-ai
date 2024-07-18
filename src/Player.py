from Maze import Maze, Cell, CellState
import math

class Player:

    def __init__(self, x: int, y: int, maze: Maze) -> None:
        self.x: int = 0
        self.y: int = 0
        self.maze: Maze = maze

    def is_walkable(self, x: int, y: int) -> bool:
        """ Check if a cell in maze is walkable given it's coordinates

        Args:
            x (int): 
            y (int): 

        Return:
            False if not walkable (wall or out of bounds)
            True if walkable (passage or coin)
        """
        if (x < 0 or x >= self.maze.size-1): return False
        if (self.maze[x][y].state == CellState.WALL): return False
        return True

    def move_left(self):
        if (is_walkable(self.x-1, self.y) == False): return
        self.x -= 1
    def move_right(self):
        if (is_walkable(self.x+1, self.y) == False): return
        self.x += 1
    def move_up(self):
        if (is_walkable(self.x, self.y+1) == False): return
        self.y += 1
    def move_down(self):
        if (is_walkable(self.x, self.y-1) == False): return
        self.x -= 1

    def get_nearest_coin(self) -> Cell:
        closest_dist = -1
        for coin in self.maze.coin_list:
            dist = math.dist([self.x, self.y], [coin.x, coin.y])
            if (closest_dist == 1):
                closest_dist = dist
                closest_coin = coin
                continue
            if (dist < closest_dist):
                closest_dist = dist
                closest_coin = coin
        return closest_coin