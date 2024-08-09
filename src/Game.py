import math
import numpy as np
from enum import Enum
from Player import Player
from Maze import Maze, Cell, CellState

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # BOMB = 4

class GameLoop:
    def __init__(self, player: Player, maze: Maze, fog_size: int = 1) -> None:
        """Constructor for GameLoop

        Args:
            player (Player): player playing the game
            maze (Maze): maze generated for the game
        """
        self.player: Player = player
        self.maze: Maze = maze
        self.reward = 0
        self.fog_size: int = fog_size
        self.visited_cells: list[Cell] = []
        self.last_visited_cell: Cell = Cell(0,0)
        self.distance_from_coin = 0
    
    def draw_maze(self) -> None:
        """Draws the player on the maze
        """
        new_maze = []
        for x in range(self.maze.size):
            row = []
            for y in range(self.maze.size):
                if x == self.player.x and y == self.player.y:
                    row.append('\033[34m#\033[0m')
                elif self.maze.grid[x][y].state == CellState.PASSAGE:
                    row.append(' ')
                elif self.maze.grid[x][y].state == CellState.WALL:
                    row.append('■')
                elif self.maze.grid[x][y].state == CellState.COIN:
                    row.append('\033[33m©\033[0m')
            new_maze.append(row)
        
        for row in new_maze:
            print(' '.join(row))

    def step(self, action):
        """Makes a step in the main game loop

        Args:
            action (_type_): _description_
        """
        # self.draw_maze()
        # print('<------------------------------------>')

        self.reward = 0
        is_done = False

        # print("Action: ", action, ", ", Action(action).name)
        if (action == Action.UP) or (action == Action.UP.value):
            if not self.player.move_up():
                self.reward -= 20
                return self.state, self.reward, is_done
        elif (action == Action.RIGHT) or (action == Action.RIGHT.value):
            if not self.player.move_right():
                self.reward -= 20
                return self.state, self.reward, is_done
        elif (action == Action.DOWN) or (action == Action.DOWN.value):
            if not self.player.move_down():
                self.reward -= 20
                return self.state, self.reward, is_done
        elif (action == Action.LEFT) or (action == Action.LEFT.value):
            if not self.player.move_left():
                self.reward -= 20
                return self.state, self.reward, is_done
        # elif (action == Action.BOMB) or (action == Action.BOMB.value):
        #     self.player.use_bomb
        # else:
        #     print('Invalid action')
        # print(f'move player to: ({self.player.x}, {self.player.y})')


        # Reward points when agent visits unvisited cells
        current_cell = self.maze.grid[self.player.x][self.player.y]
        if current_cell.visited == False and current_cell.state != CellState.WALL:
                self.reward += 30
                current_cell.visited = True
                self.visited_cells.append(current_cell)
        else:
            self.reward -= 1

        # Punishment for each step
        self.reward -= 0.5

        # if self.player.x == self.last_visited_cell.x and self.player.y == self.last_visited_cell.y: 
        #     self.reward -= 1

        if self.player.touching_coin() == True:
            self.reward += 100
        else:
            # Subtract points when agent gets further from closest coin
            nearest = self.player.get_nearest_coin()
            dist = self.player.get_distance_from_coin(nearest)
            self.reward -= dist * 0.01

        if self.player.all_coins_collected():
            print("All coins collected")
            self.reward += 300
            is_done = True

        if is_done == False:
            self.state = self.get_state()

        self.last_visited_cell: Cell = self.maze.grid[self.player.x][self.player.y]
        return self.state, self.reward, is_done

    def get_state(self) -> np.array:
        """State at step t, what the player is aware of

        Returns:
            np.array: state
        """
        state = [
            self.player.x, 
            self.player.y,
            int(self.player.all_coins_collected()),
            self.player.get_nearest_coin().x - self.player.x,
            self.player.get_nearest_coin().y - self.player.y,
            self.player.get_nearest_coin().state.value,
            self.last_visited_cell.x,
            self.last_visited_cell.y,
            self.player.get_distance_from_coin(self.player.get_nearest_coin())
        ]
        fog = self.maze.generate_fog(self.player.x, self.player.y, self.fog_size)
        for cell in fog:
            state.append(cell.x - self.player.x)
            state.append(cell.y - self.player.y)
            state.append(cell.state.value)
            # state.append(int(cell.visited))

        # Fill the remaining slots if there are
        actual_fog_size = 9 if self.fog_size == 1 else (self.fog_size*2+1)**2
        for _ in range(actual_fog_size - len(fog)):
            state.extend([-1, -1, -1])
            
        max_visited_cells = len(self.maze.get_passages())
        visited_cells = self.visited_cells[:max_visited_cells]
        for cell in visited_cells:
            state.append(cell.x - self.player.x)
            state.append(cell.y - self.player.y)
            state.append(cell.visited)
        
        # Fill the remaining slots if there are
        for _ in range(max_visited_cells - len(visited_cells)):
            state.extend([-1, -1, -1])

        return np.array(state)
                
    def reset(self, seed: int = None) -> list[int]:
        """Rests the maze to initial step

        Args:
            seed (int): seed of the maze

        Returns:
            list[int]: State of the game
        """
        self.reward = 0
        self.player.x = 0
        self.player.y = 0
        self.visited_cells = []
        self.last_visited_cell = Cell(0, 0)
        self.maze = Maze(self.maze.size, self.maze.coin_amount, seed)
        self.player.maze = self.maze
        self.state = self.get_state()

        return self.state