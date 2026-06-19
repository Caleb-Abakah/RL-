"""
environment.py


Custom Frozen Lake environments, implemented from first principles.

Two classes are provided:
    FrozenLakeEnv          - deterministic transitions
    SlipperyFrozenLakeEnv  - stochastic ("slippery") transitions

State representation
--------------------
State is a single integer in [0, n_states - 1], where:
    state = row * n_cols + col

Actions
-------
    0 = Left, 1 = Down, 2 = Right, 3 = Up
"""

import numpy as np

# The standard 8x8 Frozen Lake map from the assignment.
#   S = Start, F = Frozen (safe), H = Hole (terminal), G = Goal (terminal)
MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]


class FrozenLakeEnv:
    """A deterministic Frozen Lake grid-world environment."""

    # action constants for readability
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    def __init__(self, grid_map=MAP):
        self.grid = [list(row) for row in grid_map]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.start_state = self._find("S")
        self.state = self.start_state
        self.done = False

    # conversions between (row, col) and integer state
    def _to_state(self, row, col):
        return row * self.n_cols + col

    def _to_rowcol(self, state):
        return divmod(state, self.n_cols)

    def _find(self, ch):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == ch:
                    return self._to_state(r, c)
        raise ValueError(f"Character {ch!r} not found in map")

    def _cell(self, state):
        r, c = self._to_rowcol(state)
        return self.grid[r][c]

    def _apply_move(self, row, col, action):
        """Return the clamped (row, col) after one deterministic move."""
        if action == self.LEFT:
            col -= 1
        elif action == self.DOWN:
            row += 1
        elif action == self.RIGHT:
            col += 1
        elif action == self.UP:
            row -= 1
        else:
            raise ValueError(f"Invalid action: {action}")
        row = max(0, min(row, self.n_rows - 1))
        col = max(0, min(col, self.n_cols - 1))
        return row, col

    # ---- required API ----
    def reset(self):
        """Reset the agent to the start and begin a new episode."""
        self.state = self.start_state
        self.done = False
        return self.state

    def step(self, action):
        """Apply an action. Returns (next_state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is over. Call reset() before step().")

        row, col = self._to_rowcol(self.state)
        row, col = self._apply_move(row, col, action)

        self.state = self._to_state(row, col)
        cell = self.grid[row][col]

        if cell == "G":
            reward, self.done = 1.0, True
        elif cell == "H":
            reward, self.done = 0.0, True
        else:
            reward, self.done = 0.0, False

        return self.state, reward, self.done, {}

    def render(self):
        """Print the grid, marking the agent's position with [ ]."""
        ar, ac = self._to_rowcol(self.state)
        print("+" + "---" * self.n_cols + "+")
        for r in range(self.n_rows):
            line = "|"
            for c in range(self.n_cols):
                ch = self.grid[r][c]
                line += f"[{ch}]" if (r, c) == (ar, ac) else f" {ch} "
            print(line + "|")
        print("+" + "---" * self.n_cols + "+")

    def get_state(self):
        """Return the agent's current integer state."""
        return self.state

    def is_terminal(self):
        """True if the current state is a Hole or the Goal."""
        return self._cell(self.state) in ("H", "G")


class SlipperyFrozenLakeEnv(FrozenLakeEnv):
    """
    Frozen Lake with stochastic transitions (bonus task).

    With probability ``p_intended`` the agent moves as intended; otherwise it
    slips to one of the two perpendicular directions (each equally likely).
    ``p_intended = 1.0`` recovers the deterministic environment.
    """

    # for each action, the two perpendicular ("slip") actions
    PERP = {
        0: (1, 3),   # Left  -> slip Down or Up
        1: (0, 2),   # Down  -> slip Left or Right
        2: (1, 3),   # Right -> slip Down or Up
        3: (0, 2),   # Up    -> slip Left or Right
    }

    def __init__(self, grid_map=MAP, p_intended=1 / 3, seed=None):
        super().__init__(grid_map)
        self.p_intended = p_intended
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        """Like the base step, but the action may slip perpendicular."""
        if self.done:
            raise RuntimeError("Episode is over. Call reset() before step().")

        if self.rng.random() < self.p_intended:
            actual = action
        else:
            actual = int(self.rng.choice(self.PERP[action]))

        row, col = self._to_rowcol(self.state)
        row, col = self._apply_move(row, col, actual)

        self.state = self._to_state(row, col)
        cell = self.grid[row][col]

        if cell == "G":
            reward, self.done = 1.0, True
        elif cell == "H":
            reward, self.done = 0.0, True
        else:
            reward, self.done = 0.0, False

        return self.state, reward, self.done, {}
