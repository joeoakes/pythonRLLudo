"""
ludo_rl_steps_userprompt_commented.py

Fully commented, teaching-friendly version of a 2-player Ludo
reinforcement learning demo using tabular Q-learning.

This script:

- Defines a small but realistic Ludo board for 2 players
- Uses a Q-learning agent (player 0) vs a random opponent (player 1)
- Prompts the user for how many steps (turns) to simulate
- Logs each step and Q-learning update to a CSV file

You can use this as a classroom / teaching example to walk students
through state (S), action (A), reward (R), and Q-learning updates.
"""

import csv
import random
from collections import defaultdict
from typing import Tuple, Dict, Any


# =====================================================================
#  Ludo Environment (full board, 2 players)
# =====================================================================

class LudoEnv:
    """
    A small but realistic Ludo environment for 2 players.

    Representation
    --------------
    Each player has 4 tokens. Each token's position is encoded as:

        - -1          : HOME (still in the yard, not on the board)
        - 0..51       : main track squares on the 52-square loop
        - 100*p+1..6  : home column squares for player p (p in {0,1})
                        step 6 means that token has finished.

    From the agent's perspective (player 0), the state we expose
    is a tuple of 8 integers:

        (my0, my1, my2, my3, opp0, opp1, opp2, opp3)

    where `my*` are player 0's tokens and `opp*` are player 1's.
    """

    # Game constants
    NUM_PLAYERS = 2
    TOKENS_PER_PLAYER = 4
    MAIN_TRACK_LEN = 52        # standard Ludo loop length
    HOME_LEN = 6               # steps in the home column

    def __init__(self):
        # Where each player ENTERS the track when leaving home on a roll of 6.
        # Player 0 starts at main index 0, player 1 at index 26
        # (roughly opposite side of the board).
        self.start_square = {0: 0, 1: 26}

        # Squares on the main track where tokens cannot be captured.
        # These correspond to the "safe" star / colored squares.
        self.safe_squares = {0, 8, 13, 21, 26, 34, 39, 47}

        # Initialize board / tokens
        self.reset()

    # ------------------------------------------------------------------
    #  State management helpers
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[int, ...]:
        """
        Reset all tokens to HOME and clear episode flags.

        Returns
        -------
        state : tuple
            Initial state from player 0's perspective:
            (my0, my1, my2, my3, opp0, opp1, opp2, opp3)
        """
        # All tokens start at HOME (-1)
        self.positions = {
            p: [-1] * self.TOKENS_PER_PLAYER for p in range(self.NUM_PLAYERS)
        }
        self.done = False       # True when someone wins
        self.winner = None      # Index of winning player (0 or 1)
        return self._get_state()

    def _get_state(self) -> Tuple[int, ...]:
        """
        Build the state tuple from player 0's perspective.
        """
        my = self.positions[0]
        opp = self.positions[1]
        return tuple(my + opp)

    # --- type checks for positions -----------------------------------------------------

    def _is_home(self, pos: int) -> bool:
        """Return True if token is still at HOME (yard)."""
        return pos == -1

    def _is_track(self, pos: int) -> bool:
        """Return True if token is somewhere on the main track loop."""
        return 0 <= pos < self.MAIN_TRACK_LEN

    def _is_home_col(self, pos: int, player: int) -> bool:
        """
        Return True if token is in the player's home column.
        Player 0: 101..106   Player 1: 201..206
        """
        base = 100 * player
        return base + 1 <= pos <= base + self.HOME_LEN

    def _home_col_step(self, pos: int, player: int) -> int:
        """Return which step in the home column a token occupies (1..6)."""
        base = 100 * player
        return pos - base

    def _home_col_pos(self, player: int, step: int) -> int:
        """
        Map a home-column step (1..6) to an absolute position code.
        """
        return 100 * player + step

    def _all_finished(self, player: int) -> bool:
        """
        Check if all tokens for this player reached final home.
        """
        base = 100 * player
        return all(pos == base + self.HOME_LEN for pos in self.positions[player])

    def _distance_from_start(self, player: int, pos: int) -> int:
        """
        Return how far along the loop this token is from that player's
        start square. This is used to figure out when we should move
        from the loop into the home column.
        """
        start = self.start_square[player]
        return (pos - start) % self.MAIN_TRACK_LEN

    # ------------------------------------------------------------------
    #  Safe flags & pretty-print state (for logging)
    # ------------------------------------------------------------------

    def _safe_flags_for_player(self, player: int):
        """
        Return a tuple of 4 flags (1/0) indicating whether each token
        of `player` is "safe" (cannot be captured).

        Criteria for a token being safe:
        - Still at HOME (-1)
        - In the home column (cannot be captured there)
        - On a safe square on the main track
        """
        flags = []
        for pos in self.positions[player]:
            if pos == -1:
                flags.append(1)
            elif self._is_home_col(pos, player):
                flags.append(1)
            elif self._is_track(pos) and pos in self.safe_squares:
                flags.append(1)
            else:
                flags.append(0)
        return tuple(flags)

    def pretty_state(self, current_player: int, dice: int) -> str:
        """
        Human-readable snapshot used in the CSV trace.

        Shows:
        - my:   the current player's 4 token positions
        - opp:  the opponent's 4 token positions
        - safe: 1/0 flags for each of current player's tokens
        - dice: current dice roll
        - pid:  current player id (0 or 1)
        """
        my = tuple(self.positions[current_player])
        opp = tuple(self.positions[1 - current_player])
        safe = self._safe_flags_for_player(current_player)
        return f"(my={my}, opp={opp}, safe={safe}, dice={dice}, pid={current_player})"

    # ------------------------------------------------------------------
    #  Core Ludo movement rules
    # ------------------------------------------------------------------

    def _move_token(self, player: int, token_idx: int, dice: int):
        """
        Move one token for `player` using `dice` according to Ludo rules.

        Returns
        -------
        new_pos : int
            New position of this token after the move (or original if no move).
        moved_out : bool
            True if the token left HOME (yard) this move.
        captured : bool
            True if we captured an opponent token.
        finished_now : bool
            True if this move made this player finish all their tokens.
        """
        pos = self.positions[player][token_idx]

        # Token already in final home (cannot move anymore)
        if self._is_home_col(pos, player) and self._home_col_step(pos, player) == self.HOME_LEN:
            return pos, False, False, False

        new_pos = pos

        # --- Case 1: Token at HOME -------------------------------------
        if self._is_home(pos):
            # You must roll a 6 to leave home
            if dice == 6:
                new_pos = self.start_square[player]
            else:
                # No movement possible
                return pos, False, False, False

        # --- Case 2: Token on main track -------------------------------
        elif self._is_track(pos):
            start = self.start_square[player]
            dist = self._distance_from_start(player, pos)
            new_dist = dist + dice

            if new_dist < self.MAIN_TRACK_LEN:
                # Still on the main track after moving
                new_pos = (start + new_dist) % self.MAIN_TRACK_LEN
            else:
                # We have gone far enough to enter the home column
                steps_into_home = new_dist - self.MAIN_TRACK_LEN + 1  # 1..HOME_LEN
                if steps_into_home > self.HOME_LEN:
                    # Overshoot the final home square → illegal move
                    return pos, False, False, False
                new_pos = self._home_col_pos(player, steps_into_home)

        # --- Case 3: Token is already in home column -------------------
        else:
            step = self._home_col_step(pos, player)
            new_step = step + dice
            if new_step > self.HOME_LEN:
                # Would overshoot the end of home column
                return pos, False, False, False
            new_pos = self._home_col_pos(player, new_step)

        # Apply the move
        self.positions[player][token_idx] = new_pos

        # --- Capture logic ---------------------------------------------
        captured = False
        # Capture only happens on main track, and never on safe squares.
        if self._is_track(new_pos) and new_pos not in self.safe_squares:
            opponent = 1 - player
            for i, opp_pos in enumerate(self.positions[opponent]):
                if opp_pos == new_pos:
                    # Send opponent token back home
                    self.positions[opponent][i] = -1
                    captured = True

        # Check if this move caused all tokens to finish
        finished_now = self._all_finished(player)

        moved_out = (pos == -1 and new_pos != -1)

        return new_pos, moved_out, captured, finished_now

    # ------------------------------------------------------------------
    #  Action legality checking
    # ------------------------------------------------------------------

    def _move_preview(self, player: int, token_idx: int, dice: int) -> bool:
        """
        Check if moving this token would result in an actual change
        without permanently modifying the game state.

        Used to decide whether an action is legal.
        """
        # Save current state
        saved_pos = [list(self.positions[0]), list(self.positions[1])]
        saved_done = self.done
        saved_winner = self.winner

        legal = False
        try:
            before = self.positions[player][token_idx]
            new_pos, _, _, _ = self._move_token(player, token_idx, dice)
            # If position changed, we consider that move legal.
            if new_pos != before:
                legal = True
        finally:
            # Restore original state regardless of what happened
            self.positions[0] = saved_pos[0]
            self.positions[1] = saved_pos[1]
            self.done = saved_done
            self.winner = saved_winner

        return legal

    def get_legal_actions(self, player: int, dice: int):
        """
        Compute the list of legal actions for `player` given `dice`.

        Actions:
            0..3 : move that token index
            4    : pass (if no tokens have a legal move)
        """
        legal = []
        for idx in range(self.TOKENS_PER_PLAYER):
            if self._move_preview(player, idx, dice):
                legal.append(idx)

        # If no token can move, we must "pass"
        if not legal:
            legal = [4]
        return legal

    # ------------------------------------------------------------------
    #  Environment step
    # ------------------------------------------------------------------

    def step(self, player: int, action: int, dice: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, Any]]:
        """
        Perform ONE move in the environment for the specified player.

        Parameters
        ----------
        player : int
            0 for agent, 1 for opponent.
        action : int
            0..3 for moving a token, 4 for pass.
        dice : int
            Current dice roll (1..6).

        Returns
        -------
        next_state : tuple
            Next state from agent's (player 0) perspective.
        reward : float
            Reward from agent's perspective based on this move.
        done : bool
            True if the game ended in this step.
        info : dict
            Extra info, currently unused.
        """
        if self.done:
            raise ValueError("Episode already finished. Call reset().")

        reward = 0.0
        info: Dict[str, Any] = {}

        moved_out = False
        captured = False
        finished = False

        # Apply movement if action is a real token index
        if action in (0, 1, 2, 3):
            _, moved_out, captured, finished = self._move_token(player, action, dice)
        # else: action == 4 is "pass" → no movement

        # --------------------------------------------------------------
        # Reward scheme from the AGENT's perspective (player 0):
        #
        # Agent's own move:
        #   +1  for leaving home
        #   +5  for capturing an opponent
        #   +10 for getting a token safely home
        #   -1  for a "wasted" move (no positive event)
        #
        # Opponent's move:
        #   -1  if opponent leaves home
        #   -5  if opponent captures agent
        #   -10 if opponent gets a token home
        #   +1  if opponent wastes a move
        # --------------------------------------------------------------
        if player == 0:
            # Agent moves
            if moved_out:
                reward += 1.0
            if captured:
                reward += 5.0
            if finished:
                reward += 10.0
            if not (moved_out or captured or finished):
                reward -= 1.0
        else:
            # Opponent moves
            if moved_out:
                reward -= 1.0
            if captured:
                reward -= 5.0
            if finished:
                reward -= 10.0
            if not (moved_out or captured or finished):
                reward += 1.0

        # If this move finished the game, mark terminal state
        if finished:
            self.done = True
            self.winner = player

        return self._get_state(), reward, self.done, info


# =====================================================================
#  Q-Learning Agent
# =====================================================================

class QLearningAgent:
    """
    Simple tabular Q-learning agent controlling player 0.

    Q-learning update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.3, num_actions: int = 5):
        # Learning rate: how strongly we move toward the target each step
        self.alpha = alpha

        # Discount factor: how much we value future rewards vs immediate
        self.gamma = gamma

        # Exploration rate: probability of choosing a random action
        self.epsilon = epsilon

        # Number of actions in this environment (4 tokens + pass)
        self.num_actions = num_actions

        # Q-table: maps state -> list of Q-values (one per action)
        self.Q = defaultdict(lambda: [0.0] * num_actions)

    def choose_action(self, state: Tuple[int, ...], legal_actions):
        """
        Choose an action using epsilon-greedy strategy over legal actions.

        With probability epsilon: pick a random legal action.
        Otherwise: pick a legal action with the highest Q-value.
        """
        # Exploration: pick a random legal action
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation: pick best Q among legal actions only
        q_values = self.Q[state]
        legal_q = [(a, q_values[a]) for a in legal_actions]
        max_q = max(v for _, v in legal_q)
        best_actions = [a for a, v in legal_q if v == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Perform the Q-learning update for a single transition:

            (s, a, r, s', done)

        Returns the old Q-value, new Q-value, TD target, and max Q(s', a')
        for logging / teaching purposes.
        """
        q_values = self.Q[state]
        q_old = q_values[action]

        # If episode ended, there is no future value
        if done:
            max_next = 0.0
            target = reward
        else:
            next_q_values = self.Q[next_state]
            max_next = max(next_q_values)
            target = reward + self.gamma * max_next

        # Standard Q-learning update rule
        q_values[action] = q_old + self.alpha * (target - q_old)
        q_new = q_values[action]

        return q_old, q_new, target, max_next


# =====================================================================
#  Main driver: prompt user, run N steps, log to CSV
# =====================================================================

def run_trace():
    """
    Ask the user for a number of steps, then:

    - Run that many alternating turns of the game
    - Player 0 = Q-learning agent
    - Player 1 = random opponent
    - Log each step to a CSV with detailed RL info.
    """
    # ---------- Prompt for number of steps ----------
    while True:
        try:
            steps_str = input("Enter number of steps (positive integer): ").strip()
            num_steps = int(steps_str)
            if num_steps <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer, e.g., 20.")

    # Build an output filename based on the number of steps
    filename = f"ludo_qlearning_trace_{num_steps}steps.csv"

    # Create environment and agent
    env = LudoEnv()
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.3, num_actions=5)

    total_steps_needed = num_steps
    global_step = 0

    current_loop = 1      # episode counter (how many games)
    step_in_loop = 0      # step index within each episode
    current_player = 0    # start with the agent on turn 0

    # Get initial state
    state = env.reset()

    # ---------- Open CSV and write header ----------
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Column list mirrors the explanation in your RL spreadsheet
        writer.writerow([
            "Loop",
            "Step",
            "Player",
            "State s",
            "Dice",
            "Legal actions",
            "Chosen action a",
            "Reward r",
            "Next state s'",
            "max_a' Q(s',a')",
            "Q_old(s,a)",
            "TD target r+γ max",
            "Q_new(s,a)",
        ])

        # ---------- Main simulation loop ----------
        while global_step < total_steps_needed:
            # If game ended, start a fresh episode
            if env.done:
                current_loop += 1
                step_in_loop = 0
                current_player = 0    # agent starts new game
                state = env.reset()

            step_in_loop += 1
            global_step += 1

            # Roll a fair six-sided die
            dice = random.randint(1, 6)

            # Get set of legal actions for this player and roll
            legal_actions = env.get_legal_actions(current_player, dice)

            # Build a human-readable string for the legal actions
            legal_names = []
            for a in legal_actions:
                if a in (0, 1, 2, 3):
                    pos = env.positions[current_player][a]
                    if pos == -1 and dice == 6:
                        legal_names.append(f"move_token_{a}_out")
                    else:
                        legal_names.append(f"move_token_{a}_by_{dice}")
                else:
                    legal_names.append("pass")
            legal_str = ", ".join(legal_names)

            # Human-readable view of the state for the CSV
            state_str = env.pretty_state(current_player, dice)

            # Choose action: agent uses epsilon-greedy, opponent is random
            if current_player == 0:
                action = agent.choose_action(state, legal_actions)
            else:
                action = random.choice(legal_actions)

            # Translate numeric action into a human-readable label
            if action in (0, 1, 2, 3):
                pos = env.positions[current_player][action]
                if pos == -1 and dice == 6:
                    action_name = f"move_token_{action}_out"
                else:
                    action_name = f"move_token_{action}_by_{dice}"
            else:
                action_name = "pass"

            # Step the environment with the chosen action
            next_state, reward, done, info = env.step(current_player, action, dice)
            next_state_str = env.pretty_state(current_player, dice)

            # Q-learning update only for agent's real moves
            if current_player == 0 and action != 4:
                q_old, q_new, target, max_next = agent.update(
                    state, action, reward, next_state, done
                )
                # Round for nicer logging
                q_old_log = round(q_old, 3)
                q_new_log = round(q_new, 3)
                target_log = round(target, 3)
                max_next_log = round(max_next, 3)
            else:
                # Opponent or pass → no Q-update
                q_old_log = q_new_log = target_log = max_next_log = ""

            # Write a single row for this step
            writer.writerow([
                current_loop,        # which game (episode) we are in
                step_in_loop,        # step index within that episode
                current_player,      # 0 = agent, 1 = opponent
                state_str,           # human-readable state
                dice,                # dice roll
                legal_str,           # legal actions
                action_name,         # chosen action
                round(reward, 3),    # scalar reward
                next_state_str,      # human-readable next state
                max_next_log,        # max_a' Q(s', a')
                q_old_log,           # Q_old(s,a)
                target_log,          # TD target
                q_new_log,           # Q_new(s,a)
            ])

            # Move on: next state becomes current state
            state = next_state
            # Alternate players: 0 → 1 → 0 → 1 ...
            current_player = 1 - current_player

    print(f"Finished {num_steps} steps. Trace saved to '{filename}'.")


# =====================================================================
#  Script entry point
# =====================================================================

if __name__ == "__main__":
    run_trace()
