import random
from collections import defaultdict

class LudoEnv:
    """
    Simplified 2-player Ludo-like environment for RL.

    - 2 players: agent (player 0) and opponent (player 1)
    - 2 tokens per player
    - Linear board: positions 0..BOARD_LEN
    - Goal: Get both of your tokens exactly to BOARD_LEN
    - If you land on an opponent token (not finished), you capture it (send to 0)
    - Agent moves, then opponent moves (random policy)
    - Step returns state from the agent's perspective only.
    """

    def __init__(self, board_len=15):
        self.board_len = board_len
        self.reset()

    def reset(self):
        # Positions: dict[player] = [token0_pos, token1_pos]
        self.positions = {
            0: [0, 0],  # agent
            1: [0, 0],  # opponent
        }
        self.done = False
        self.winner = None
        return self._get_state()

    def _get_state(self):
        # State is always from agent's perspective
        # (p0_t0, p0_t1, p1_t0, p1_t1)
        return (
            self.positions[0][0],
            self.positions[0][1],
            self.positions[1][0],
            self.positions[1][1],
        )

    def _move_token(self, player, token_idx, dice):
        """
        Move a single token if possible.
        Returns:
            captured (bool), finished_token_now (bool)
        """
        pos = self.positions[player][token_idx]

        # Token already finished
        if pos == self.board_len:
            return False, False

        # Try to move
        new_pos = pos + dice
        if new_pos > self.board_len:
            # Overshoot: can't move
            return False, False

        self.positions[player][token_idx] = new_pos
        captured = False

        # Capture logic: only if token did not just finish
        if new_pos < self.board_len:
            opponent = 1 - player
            for i, opp_pos in enumerate(self.positions[opponent]):
                if opp_pos == new_pos:
                    # Capture: send opponent token back to start
                    self.positions[opponent][i] = 0
                    captured = True

        finished_now = (self.positions[player][0] == self.board_len and
                        self.positions[player][1] == self.board_len)
        return captured, finished_now

    def _opponent_move(self):
        """
        Opponent plays one random move (if possible).
        Returns:
            opponent_won (bool)
        """
        dice = random.randint(1, 6)

        # Find tokens that COULD move
        movable = []
        for idx, pos in enumerate(self.positions[1]):
            if pos == self.board_len:
                continue
            if pos + dice <= self.board_len:
                movable.append(idx)

        if not movable:
            # No legal move
            return False

        token_choice = random.choice(movable)
        _, finished_now = self._move_token(1, token_choice, dice)
        return finished_now

    def step(self, action):
        """
        Perform one agent decision step.

        Parameters:
            action: 0 or 1 (which of agent's tokens to try to move)

        Returns:
            next_state, reward, done, info
        """
        if self.done:
            raise ValueError("Episode already done. Call reset().")

        reward = 0.0
        info = {}

        # --- Agent move ---
        dice = random.randint(1, 6)
        captured, finished_now = self._move_token(0, action, dice)

        # Small reward shaping
        if captured:
            reward += 0.3  # reward for capturing
        reward -= 0.01    # small step penalty to encourage faster wins

        if finished_now:
            # Agent wins the game
            reward += 1.0
            self.done = True
            self.winner = 0
            info["winner"] = "agent"
            return self._get_state(), reward, self.done, info

        # --- Opponent move (environment dynamics) ---
        opponent_won = self._opponent_move()

        if opponent_won:
            reward -= 1.0  # losing is bad
            self.done = True
            self.winner = 1
            info["winner"] = "opponent"

        return self._get_state(), reward, self.done, info


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.2, num_actions=2):
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # exploration rate
        self.num_actions = num_actions
        # Q[state][action] = value
        self.Q = defaultdict(lambda: [0.0] * self.num_actions)

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.Q[state]
        max_q = max(q_values)
        # In case of tie, randomly choose among best actions
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Standard Q-learning update."""
        q_values = self.Q[state]
        q_sa = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self.Q[next_state]
            target = reward + self.gamma * max(next_q_values)

        q_values[action] = q_sa + self.alpha * (target - q_sa)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


def train_ludo_agent(
    num_episodes=5000,
    board_len=15,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=0.5,
    epsilon_end=0.05
):
    env = LudoEnv(board_len=board_len)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start)

    wins = 0
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        # Linearly decay epsilon
        frac = episode / num_episodes
        eps = epsilon_start + frac * (epsilon_end - epsilon_start)
        agent.set_epsilon(eps)

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            if done and info.get("winner") == "agent":
                wins += 1

        if episode % 500 == 0:
            win_rate = wins / episode
            print(f"Episode {episode}/{num_episodes}, "
                  f"epsilon={agent.epsilon:.3f}, "
                  f"win rate so far={win_rate:.3f}")

    print("Training finished.")
    print(f"Final win rate over {num_episodes} episodes: {wins/num_episodes:.3f}")
    return env, agent


def play_one_game(env, agent, verbose=True):
    """
    Let the trained agent play one game (epsilon=0, greedy policy).
    """
    agent_epsilon_backup = agent.epsilon
    agent.set_epsilon(0.0)  # purely greedy

    state = env.reset()
    done = False
    step_count = 0

    if verbose:
        print("\n=== Demo game with trained agent ===")
        print("State format: (p0_t0, p0_t1, p1_t0, p1_t1)")

    while not done:
        step_count += 1
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        if verbose:
            print(f"Step {step_count}:")
            print(f"  State: {state}")
            print(f"  Agent chooses token: {action}")
            print(f"  Next state: {next_state}, reward={reward:.2f}")

        state = next_state

    if verbose:
        print("Game over.")
        print("Winner:", "Agent" if env.winner == 0 else "Opponent")

    agent.set_epsilon(agent_epsilon_backup)


if __name__ == "__main__":
    # 1. Train the agent
    env, agent = train_ludo_agent(
        num_episodes=5000,
        board_len=15,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.05
    )

    # 2. Run a demo game with the trained agent
    play_one_game(env, agent, verbose=True)
