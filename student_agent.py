# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import json
from collections import defaultdict


def rot90(pattern):
    return [(y, 3 - x) for x, y in pattern]

def rot180(pattern):
    return [(3 - x, 3 - y) for x, y in pattern]

def rot270(pattern):
    return [(3 - y, x) for x, y in pattern]

def reflect(pattern):
    return [(x, 3 - y) for x, y in pattern]

def print_board(board):
    for row in board:
        print("\t".join(str(num) if num != 0 else '.' for num in row))


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid
        before_add = copy.deepcopy(self.board)

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, before_add

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = {}
        for pattern in self.patterns:
            self.symmetry_patterns[pattern] = self.generate_symmetries(pattern)
            #print(self.symmetry_patterns[pattern])

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        symmetries = set()
        symmetries.add(tuple(pattern))
        #print(tuple(pattern))
        symmetries.add(tuple(rot90(pattern)))
        #print(tuple(rot90(pattern)))
        symmetries.add(tuple(rot180(pattern)))
        #print(tuple(rot180(pattern)))
        symmetries.add(tuple(rot270(pattern)))
        #print(tuple(rot270(pattern)))
        reflected_pattern = reflect(pattern)
        symmetries.add(tuple(reflected_pattern))
        #print(tuple(reflected_pattern))
        symmetries.add(tuple(rot90(reflected_pattern)))
        #print(tuple(rot90(reflected_pattern)))
        symmetries.add(tuple(rot180(reflected_pattern)))
        #print(tuple(rot180(reflected_pattern)))
        symmetries.add(tuple(rot270(reflected_pattern)))
        #print(tuple(rot270(reflected_pattern)))
        #print(list(symmetries))
        return list(symmetries)



    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        feature = []
        for x, y in coords:
            feature.append(self.tile_to_index(board[x][y]))
        #print(tuple(feature))
        return tuple(feature)


    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0.0
        for pattern, weight in zip(self.symmetry_patterns, self.weights):
            for sym_pattern in self.symmetry_patterns[pattern]:
              feature = self.get_feature(board, sym_pattern)
              total_value += weight[feature]
              #print(feature, weight[feature])
        #print('total value:',total_value)
        return total_value

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for pattern, weight in zip(self.symmetry_patterns, self.weights):
            for sym_pattern in self.symmetry_patterns[pattern]:
                feature = self.get_feature(board, sym_pattern)
                weight[feature] += alpha * delta / 8

class TD_MCTS_Node:
    def __init__(self, state, parent=None, action=None, is_afterstate=False):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.is_afterstate = is_afterstate
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        if not is_afterstate:
            env = Game2048Env()
            env.board = state
            self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        else:
            self.untried_actions = []

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        return len([a for a in range(4) if env.is_move_legal(a)]) == 0

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99, V_norm=100):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.V_norm = V_norm

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        if node.is_afterstate:
            children = list(node.children.items())
            probs = [child.prob for _, child in children]
            selected = random.choices(children, weights=probs, k=1)[0]
            return selected[1]
        else:
            return max(
                node.children.values(),
                key=lambda child: (child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits) / child.visits))
            )

    def evaluate(self, env):
        best_value = -float('inf')
        for action in range(4):
            if env.is_move_legal(action):
                sim_env = copy.deepcopy(env)
                next_state, reward, done, after_state = sim_env.step(action)
                value = self.approximator.value(after_state) + reward
                best_value = max(best_value, value)
        return best_value

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        current_depth = 0
        value = 0
        while current_depth < depth:
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            state, reward, done, _ = sim_env.step(action)
            current_depth += 1
            value = approximator.value(state)

        return value


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def expand(self, node, sim_env):
        # Expand all legal player moves (state -> afterstate)
        if not node.is_afterstate:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            next_state, reward, done, after_state = sim_env.step(action)
            sim_env.board = after_state
            after_node = TD_MCTS_Node(
                state=after_state,
                parent=node,
                action=action,
                is_afterstate=True
            )
            after_node.total_reward = self.approximator.value(after_state)
            node.children[action] = after_node

            node = after_node

        empty_cells = [(i, j) for i in range(4) for j in range(4) if node.state[i][j] == 0]
        if not empty_cells:
            return node
        rand_cell = random.choice(empty_cells)
        i, j = rand_cell

        if random.random() < 0.9:
            sim_env.board[i][j] = 2
        else:
            sim_env.board[i][j] = 4

        if (i, j, sim_env.board[i][j]) not in node.children.keys():
            child_node = TD_MCTS_Node(
                state=sim_env.board,
                parent=node,
                action=None,
                is_afterstate=False,
            )
            node.children[(i, j, sim_env.board[i][j])] = child_node

        return node.children[(i, j, sim_env.board[i][j])]


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, 0)

        while node.fully_expanded() and node.children:
            best_node = max(node.children.values(), key=lambda child: (child.total_reward / (child.visits + 1e-6)) + self.c * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-6)))
            next_state, reward, done, afterstate = sim_env.step(best_node.action)
            sim_env.board = afterstate
            node = best_node
            node = self.expand(node, sim_env)

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions:
            node = self.expand(node, sim_env)
        
        value = self.evaluate(sim_env)
        norm_value = value / self.V_norm

        # Rollout: Simulate a random game from the expanded node.
        # rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, norm_value)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        best_value = -float('inf')
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits or (child.visits == best_visits and child.total_reward > best_value):
                best_visits = child.visits
                best_action = action
        return best_action, distribution


def hex_to_tuple(s: str):
    hex_number = hex(int(s))
    hex_number = hex_number[2:]
    hex_number = "0" * (6 - len(hex_number)) + hex_number
    reversed_hex = hex_number[::-1]
    decimal_values = []
    for char in reversed_hex:
        decimal_value = int(char, 16)
        decimal_values.append(decimal_value)

    return tuple(decimal_values)

patterns = [
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
    ((1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)),
    ((2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

file_path = "weights-new.json"
with open(file_path, "r") as file:
    weight_ = json.load(file)


for pid, wt in weight_.items():
    for s, w in wt.items():
        t = hex_to_tuple(s)
        approximator.weights[int(pid)][t] = w
        #if not int(pid) and int(s) < 10:
            #print(t, w)

env = Game2048Env()
td_mcts = TD_MCTS(env, approximator, iterations=135, exploration_constant=1.41, gamma=1, V_norm=10000)
c = 0

def get_action(state, score):
    #env = Game2048Env()
    #return random.choice([0, 1, 2, 3]) # Choose a random action
    global c
    best_action = None
    if not c:
        for i in range(4):
            for j in range(4):
                if state[i][j]>=8192:
                    c = 1
    
        env.board = state.copy()
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        action_values = []
        for move in legal_moves:
            #print('env')
            #print_board(env.board)
            temp_env = copy.deepcopy(env)
            next_state, new_score, done, before_add = temp_env.step(move)
            #print('temp')
            #print_board(temp_env.board)
            value = approximator.value(before_add)
            action_values.append((move, value))
        best_action = max(action_values, key=lambda x: x[1])[0]
        return best_action

    else:
        root = TD_MCTS_Node(state)

        # Run multiple simulations to build the MCTS tree
        for _ in range(td_mcts.iterations):
            td_mcts.run_simulation(root)

        # Select the best action (based on highest visit count)
        best_action, _ = td_mcts.best_action_distribution(root)

        return best_action
        
    # You can submit this random agent to evaluate the performance of a purely random strategy.


