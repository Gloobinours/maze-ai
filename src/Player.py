from Maze import Maze, Cell, CellState
import math

class Player:

    def __init__(self, x: int, y: int, maze: Maze) -> None:
        """Constructor for Player

        Args:
            x (int): x coordinate of the player
            y (int): y coordinate of the player
            maze (Maze): the maze the player is playing on
        """
        self.x: int = x
        self.y: int = y
        self.maze: Maze = maze

    def is_walkable(self, x: int, y: int) -> bool:
        """ Check if a cell in the maze is walkable given its coordinates

        Args:
            x (int): The x-coordinate of the cell
            y (int): The y-coordinate of the cell

        Returns:
            bool: False if the cell is not walkable (wall or out of bounds),
                True if the cell is walkable (passage or coin)
        """
        if (y < 0 or y >= self.maze.size): 
            print(f"Out of bounds: ({x}, {y})")
            return False
        if (x < 0 or x >= self.maze.size): 
            print(f"Out of bounds: ({x}, {y})")
            return False
        
        if self.maze.grid[x][y].state == CellState.WALL:
            # print(f"Wall at: ({x}, {y})")
            return False
        
        # The cell is walkable
        return True

    def move_left(self) -> bool:
        """Move player left
        """
        if (self.is_walkable(self.x, self.y-1)):
            self.y -= 1
            return True
        else:
            return False
    def move_right(self) -> bool:
        """Move player right
        """
        if (self.is_walkable(self.x, self.y+1)):
            self.y += 1
            return True
        else:
            return False
    def move_up(self) -> bool:
        """Move player up
        """
        if (self.is_walkable(self.x-1, self.y)):
            self.x -= 1
            return True
        else:
            return False
    def move_down(self) -> bool:
        """Move player down
        """
        if (self.is_walkable(self.x+1, self.y)):
            self.x += 1
            return True
        else:
            return False

    def get_nearest_coin(self) -> Cell:
        """Get the position of the closest coin from player

        Returns:
            Cell: closest coin from player
        """
        closest_dist = float('inf')
        closest_coin = None

        for coin in self.maze.coin_list:
            dist = math.dist([self.x, self.y], [coin.x, coin.y])
            if (dist < closest_dist):
                closest_dist = dist
                closest_coin = coin
                
        return closest_coin
    
    def get_distance_from_coin(self, coin: Cell):
        return math.dist([self.x, self.y], [coin.x, coin.y])
    
    def all_coins_collected(self) -> bool:
        """Check if all the coins are collected by the player

        Returns:
            bool: True if all coins are collected
        """
        return len(self.maze.coin_list) == 0
    
    def touching_coin(self) -> bool:
        """Delete a coin when the player touches it
        """
        if self.maze.grid[self.x][self.y].state == CellState.COIN:
            self.maze.delete_coin(self.x, self.y)
            return True
    
    def use_bomb(self):
        """Use bomb to break walls
        """
        self.maze.explode_bomb(self.x, self.y)
