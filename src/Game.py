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
        self.steps = 0
        self.truncated = False
    
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
        self.steps += 1
        self.reward = 0
        is_done = False

        # print("Action: ", action, ", ", Action(action).name)
        if (action == Action.UP) or (action == Action.UP.value):
            if not self.player.move_up():
                self.reward -= 20
                self.truncated = True
                return self.state, self.reward, is_done, self.truncated
        elif (action == Action.RIGHT) or (action == Action.RIGHT.value):
            if not self.player.move_right():
                self.reward -= 20
                self.truncated = True
                return self.state, self.reward, is_done, self.truncated
        elif (action == Action.DOWN) or (action == Action.DOWN.value):
            if not self.player.move_down():
                self.reward -= 20
                self.truncated = True
                return self.state, self.reward, is_done, self.truncated
        elif (action == Action.LEFT) or (action == Action.LEFT.value):
            if not self.player.move_left():
                self.reward -= 20
                self.truncated = True
                return self.state, self.reward, is_done, self.truncated
        # elif (action == Action.BOMB) or (action == Action.BOMB.value):
        #     self.player.use_bomb
        else:
            print('Invalid action')

        # Reward points when agent visits unvisited cells
        current_cell = self.maze.grid[self.player.x][self.player.y]
        if current_cell.visited == False and current_cell.state != CellState.WALL:
                self.reward += 10
                current_cell.visited = True
                self.visited_cells.append(current_cell)
        # else:
        #     self.reward -= 1

        # Punishment for each step
        self.reward -= 0.5

        # if self.player.x == self.last_visited_cell.x and self.player.y == self.last_visited_cell.y: 
        #     self.reward -= 1

        if self.player.touching_coin() == True:
            self.reward += 500
        # else:
        #     # Subtract points when agent gets further from closest coin
        #     nearest = self.player.get_nearest_coin()
        #     dist = self.player.get_distance_from_coin(nearest)
        #     self.reward -= dist * 0.0

        if self.player.all_coins_collected():
            print("All coins collected")
            self.reward += 1000
            is_done = True

        if is_done == False:
            self.state = self.get_state()

        self.last_visited_cell: Cell = self.maze.grid[self.player.x][self.player.y]
        return self.state, self.reward, is_done, self.truncated

    def get_state(self) -> np.array:
        """State at step t, what the player is aware of

        Returns:
            np.array: state
        """
        state = [
            # self.player.x, #
            # self.player.y, #
            # self.player.get_nearest_coin().x - self.player.x,
            # self.player.get_nearest_coin().y - self.player.y,
            # self.player.get_nearest_coin().state.value,
            # self.last_visited_cell.x,
            # self.last_visited_cell.y,
            # self.player.get_distance_from_coin(self.player.get_nearest_coin()),
            # self.steps
        ]
        rays = self.player.get_rays()
        for length in rays['lengths']:
            state.append(length)
        # for count in rays['univisted_cnt']:
        #     state.append(count)
        for coin in rays['touches_coin']:
            state.append(coin)

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