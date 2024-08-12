import random
from enum import Enum

class CellState(Enum):
    """State of a cell

    Args:
        Enum (int): Each number is a state
    """
    PASSAGE = 0
    WALL = 1
    COIN = 2

class Cell:
    def __init__(self, x: int, y: int) -> None:
        """Cell Constructor

        Args:
            x (int): x coordinaate
            y (int): y coordinate
        """
        self.x: int = x
        self.y: int = y
        self.state: CellState = CellState.WALL
        self.visited = False

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y
    def __str__(self) -> str:
        return f"[({self.x}, {self.y}), {self.state}]"


class Maze:
    def __init__(self, size: int, coin_amount : int, a_seed : int = None) -> None:
        """Constructor for Maze

        Args:
            size (int): The size of the maze (it will be a square maze)
            coin_amount (int): The number of coins to place in the maze
        """
        self.size: int = size
        if a_seed: random.seed(a_seed)
        self.grid = [[Cell(x, y) for y in range(size)] for x in range(size)]
        self.generate_grid()

        self.coin_amount: int = coin_amount
        self.coin_list = []
        self.add_coins_to_maze()

    def get_neighbors(self, cell: Cell, matrix) -> None:
        """Get neighboring cells, 2 cells away from passed in cell

        Args:
            cell (Cell): current cell
            matrix (_type_): maze grid

        Returns:
            list: list of cells
        """
        directions = [(-2, 0), (2,0), (0, -2), (0, 2)]
        neighbors = []
        for d in directions:
            new_row = cell.x + d[0]
            new_col = cell.y + d[1]

            if (0 <= new_row < self.size) and (0 <= new_col < self.size):
                neighbors.append(matrix[new_row][new_col])

        return neighbors
    
    def get_passages(self) -> list[Cell]:
        passages = []
        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if (self.grid[x][y].state == CellState.PASSAGE):
                    passages.append(self.grid[x][y])

        return passages

    def generate_grid(self) -> list:
        """Generate the maze grid using DFS

        Returns:
            list: returns a matrix of cells
        """
        maze = self.grid
        # Choose the initial cell, mark it as visited and push it to the stack
        random_cell = maze[0][0]
        random_cell.state = CellState.PASSAGE
        visited = [random_cell]
        stack = [random_cell]

        while stack:
            current_cell = stack.pop()
            # List of unvisited neighbors
            unvisited_neighbors :list = [n for n in self.get_neighbors(current_cell, maze) if n not in visited]
            # If the current cell has any neighbours which have not been visited
            if unvisited_neighbors:
                stack.append(current_cell)
                # Choose one of the unvisited neighbours
                chosen_cell = random.choice(unvisited_neighbors)
                # Remove the wall between the current cell and the chosen cell
                in_between_x = current_cell.x + (chosen_cell.x - current_cell.x) // 2
                in_between_y = current_cell.y + (chosen_cell.y - current_cell.y) // 2
                in_between = maze[in_between_x][in_between_y]
                in_between.state = CellState.PASSAGE
                chosen_cell.state = CellState.PASSAGE
                # Mark the chosen cell as visited and push it to the stack
                visited.append(chosen_cell)
                stack.append(chosen_cell)
        return maze

    def check_adjacent(self, row: int, col: int) -> bool:
        """return true if excatly 3 adjacent wall to point

        Args:
            row (int): the row of the maze's grid
            col (int): the column of the maze's grid

        Returns:
            bool: false if not exactly 3 walls surounding
        """
        rows = len(self.grid)
        cols = len(self.grid[0])

        directions = [(-1, 0), (1,0), (0, -1), (0, 1)]

        count_walls = 0

        for d in directions:
            new_row = row + d[0]
            new_col = col + d[1]

            if (0 <= new_row < rows) and (0 <= new_col < cols) and (self.grid[new_row][new_col].state == CellState.WALL):
                count_walls += 1
        
        return count_walls == 3
    
    def generate_coins(self) -> list:
        """Generate coins when 3 walls around

        Returns:
            list: list of random possible coin locations
        """
        possible_points = []
        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if (self.check_adjacent(x, y) and self.grid[x][y].state == CellState.PASSAGE):
                    possible_points.append((x,y))

        if len(possible_points) == 0:
            x = random.randrange(0, self.size)
            y = random.randrange(0, self.size)
            while self.grid[x][y].state == CellState.WALL:
                x = random.randrange(0, self.size)
                y = random.randrange(0, self.size)
            return [(x,y)]
                    
        if len(possible_points) < self.coin_amount:
            self.coin_amount = len(possible_points)

        return random.sample(possible_points, self.coin_amount)
    
    def add_coins_to_maze(self) -> None:
        """Append coins to matrix
        """
        coin_pos = self.generate_coins()
        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                for pos in coin_pos:
                    if (x == pos[0] and y == pos[1]):
                        self.grid[x][y].state = CellState.COIN
                        self.coin_list.append(self.grid[x][y])

    def explode_bomb(self, x, y) -> None:
        """Break walls around bomb when touched

        Args:
            x, y (int): cell coordinates of player
        """
        
        for i in range(x-1,x+2):
            for j in range(y-1,y+2):
                if (i >= 0 or j >= 0) and (i <= self.size or j <= self.size):
                    if self.grid[i][j].state == CellState.WALL:
                        self.grid[i][j].state = CellState.PASSAGE

    def generate_fog(self, x: int, y: int, size: int):
        """Generate the fog of war of the maze

        Args:
            x (int): X-Coordinate of the player's position
            y (int): Y-Coordinate of the player's position
            size (int): Size of the visible area around the player

        Returns:
            list[Cell]: Clear area around the player of certain size 
        """
        res = []
        for i in range(x - size, x + size + 1):
            for j in range(y - size, y + size + 1):
                if (i >= 0 and j >= 0) and (i < self.size and j < self.size):
                    res.append(self.grid[i][j])

        return res

    def delete_coin(self, x, y) -> None:
        """Delete coin when player touches it

        Args:
            x (int): x coordinate of the player
            y (int): y coordinate of the player
        """
        coin_cell = self.grid[x][y]
        coin_cell.state = CellState.PASSAGE
        if coin_cell in self.coin_list:
            self.coin_list.remove(coin_cell)
    
    def __str__(self) -> str:
        return '\n'.join(' '.join(str(cell) for cell in row) for row in self.grid)
