
"""
ludo_rl_steps_userprompt.py

Two-Player Ludo — User-Defined Number of Steps — Tabular Q-Learning Trace
-------------------------------------------------------------------------

This script defines a small but reasonably realistic **2-player Ludo game**
and runs **tabular Q-learning** on it for a user-specified number of steps.

It then writes a detailed **CSV trace** showing each reinforcement learning
update.

See in-code comments for details.
"""

import csv
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any


# =====================================================================
#  Ludo Environment (full board, 2 players)
# =====================================================================

class LudoEnv:
    """A small but realistic Ludo environment for 2 players."""

    NUM_PLAYERS = 2
    TOKENS_PER_PLAYER = 4
    MAIN_TRACK_LEN = 52
    HOME_LEN = 6

    def __init__(self):
        self.start_square = {0: 0, 1: 26}
        self.safe_squares = {0, 8, 13, 21, 26, 34, 39, 47}
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        self.positions = {
            p: [-1] * self.TOKENS_PER_PLAYER for p in range(self.NUM_PLAYERS)
        }
        self.done = False
        self.winner = None
        return self._get_state()

    def _get_state(self) -> Tuple[int, ...]:
        my = self.positions[0]
        opp = self.positions[1]
        return tuple(my + opp)

    def _is_home(self, pos: int) -> bool:
        return pos == -1

    def _is_track(self, pos: int) -> bool:
        return 0 <= pos < self.MAIN_TRACK_LEN

    def _is_home_col(self, pos: int, player: int) -> bool:
        base = 100 * player
        return base + 1 <= pos <= base + self.HOME_LEN

    def _home_col_step(self, pos: int, player: int) -> int:
        base = 100 * player
        return pos - base

    def _home_col_pos(self, player: int, step: int) -> int:
        return 100 * player + step

    def _all_finished(self, player: int) -> bool:
        base = 100 * player
        return all(pos == base + self.HOME_LEN for pos in self.positions[player])

    def _distance_from_start(self, player: int, pos: int) -> int:
        start = self.start_square[player]
        return (pos - start) % self.MAIN_TRACK_LEN

    def _safe_flags_for_player(self, player: int) -> Tuple[int, int, int, int]:
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
        my = tuple(self.positions[current_player])
        opp = tuple(self.positions[1 - current_player])
        safe = self._safe_flags_for_player(current_player)
        return f"(my={my}, opp={opp}, safe={safe}, dice={dice}, pid={current_player})"

    def _move_token(self, player: int, token_idx: int, dice: int):
        pos = self.positions[player][token_idx]

        if self._is_home_col(pos, player) and self._home_col_step(pos, player) == self.HOME_LEN:
            return pos, False, False, False

        new_pos = pos

        if self._is_home(pos):
            if dice == 6:
                new_pos = self.start_square[player]
            else:
                return pos, False, False, False
        elif self._is_track(pos):
            start = self.start_square[player]
            dist = self._distance_from_start(player, pos)
            new_dist = dist + dice

            if new_dist < self.MAIN_TRACK_LEN:
                new_pos = (start + new_dist) % self.MAIN_TRACK_LEN
            else:
                steps_into_home = new_dist - self.MAIN_TRACK_LEN + 1
                if steps_into_home > self.HOME_LEN:
                    return pos, False, False, False
                new_pos = self._home_col_pos(player, steps_into_home)
        else:
            step = self._home_col_step(pos, player)
            new_step = step + dice
            if new_step > self.HOME_LEN:
                return pos, False, False, False
            new_pos = self._home_col_pos(player, new_step)

        self.positions[player][token_idx] = new_pos
        captured = False

        if self._is_track(new_pos) and new_pos not in self.safe_squares:
            opponent = 1 - player
            for i, opp_pos in enumerate(self.positions[opponent]):
                if opp_pos == new_pos:
                    self.positions[opponent][i] = -1
                    captured = True

        finished_now = self._all_finished(player)
        moved_out = (pos == -1 and new_pos != -1)
        return new_pos, moved_out, captured, finished_now

    def _move_preview(self, player: int, token_idx: int, dice: int) -> bool:
        saved_pos = [list(self.positions[0]), list(self.positions[1])]
        saved_done = self.done
        saved_winner = self.winner

        legal = False
        try:
            before = self.positions[player][token_idx]
            new_pos, moved_out, captured, finished = self._move_token(player, token_idx, dice)
            if new_pos != before:
                legal = True
        finally:
            self.positions[0] = saved_pos[0]
            self.positions[1] = saved_pos[1]
            self.done = saved_done
            self.winner = saved_winner

        return legal

    def get_legal_actions(self, player: int, dice: int):
        legal = []
        for idx in range(self.TOKENS_PER_PLAYER):
            if self._move_preview(player, idx, dice):
                legal.append(idx)
        if not legal:
            legal = [4]
        return legal

    def step(self, player: int, action: int, dice: int):
        if self.done:
            raise ValueError("Episode already finished. Call reset().")

        reward = 0.0
        info: Dict[str, Any] = {}

        moved_out = False
        captured = False
        finished = False

        if action in (0, 1, 2, 3):
            _, moved_out, captured, finished = self._move_token(player, action, dice)

        if player == 0:
            if moved_out:
                reward += 1.0
            if captured:
                reward += 5.0
            if finished:
                reward += 10.0
            if not (moved_out or captured or finished):
                reward -= 1.0
        else:
            if moved_out:
                reward -= 1.0
            if captured:
                reward -= 5.0
            if finished:
                reward -= 10.0
            if not (moved_out or captured or finished):
                reward += 1.0

        if finished:
            self.done = True
            self.winner = player
            info["winner"] = player

        return self._get_state(), reward, self.done, info


# =====================================================================
#  Q-Learning Agent
# =====================================================================

class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.3, num_actions: int = 5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.Q = defaultdict(lambda: [0.0] * num_actions)

    def choose_action(self, state: Tuple[int, ...], legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        q_values = self.Q[state]
        legal_q = [(a, q_values[a]) for a in legal_actions]
        max_q = max(v for _, v in legal_q)
        best_actions = [a for a, v in legal_q if v == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        q_values = self.Q[state]
        q_old = q_values[action]

        if done:
            max_next = 0.0
            target = reward
        else:
            next_q_values = self.Q[next_state]
            max_next = max(next_q_values)
            target = reward + self.gamma * max_next

        q_values[action] = q_old + self.alpha * (target - q_old)
        q_new = q_values[action]
        return q_old, q_new, target, max_next


# =====================================================================
#  Main driver
# =====================================================================

def run_trace():
    while True:
        try:
            steps_str = input("Enter number of steps (positive integer): ").strip()
            num_steps = int(steps_str)
            if num_steps <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer, e.g., 20.")

    filename = f"ludo_qlearning_trace_{num_steps}steps.csv"

    env = LudoEnv()
    agent = QLearningAgent()

    total_steps_needed = num_steps
    global_step = 0

    current_loop = 1
    step_in_loop = 0
    current_player = 0
    state = env.reset()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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

        while global_step < total_steps_needed:
            if env.done:
                current_loop += 1
                step_in_loop = 0
                current_player = 0
                state = env.reset()

            step_in_loop += 1
            global_step += 1

            dice = random.randint(1, 6)
            legal_actions = env.get_legal_actions(current_player, dice)

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

            state_str = env.pretty_state(current_player, dice)

            if current_player == 0:
                action = agent.choose_action(state, legal_actions)
            else:
                action = random.choice(legal_actions)

            if action in (0, 1, 2, 3):
                pos = env.positions[current_player][action]
                if pos == -1 and dice == 6:
                    action_name = f"move_token_{action}_out"
                else:
                    action_name = f"move_token_{action}_by_{dice}"
            else:
                action_name = "pass"

            next_state, reward, done, info = env.step(current_player, action, dice)
            next_state_str = env.pretty_state(current_player, dice)

            if current_player == 0 and action != 4:
                q_old, q_new, target, max_next = agent.update(
                    state, action, reward, next_state, done
                )
                q_old_log = round(q_old, 3)
                q_new_log = round(q_new, 3)
                target_log = round(target, 3)
                max_next_log = round(max_next, 3)
            else:
                q_old_log = q_new_log = target_log = max_next_log = ""

            writer.writerow([
                current_loop,
                step_in_loop,
                current_player,
                state_str,
                dice,
                legal_str,
                action_name,
                round(reward, 3),
                next_state_str,
                max_next_log,
                q_old_log,
                target_log,
                q_new_log,
            ])

            state = next_state
            current_player = 1 - current_player

    print(f"Finished {num_steps} steps. Trace saved to '{filename}'.")


if __name__ == "__main__":
    run_trace()
