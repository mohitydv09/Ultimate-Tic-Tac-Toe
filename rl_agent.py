import torch
from collections import deque
import random
from typing import List, Tuple

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class RLAgent:
    def __init__(self, sign = "X"):
        self.num_games = 0
        self.epsilon = 0 ## Exploration rate
        self.gamma = 0.9 ## Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None ## TODO
        self.trainer = None ## TODO
        self.sign = sign

    def get_state(self, state : "UtttState") -> torch.tensor:
        curr_player = state.current.sign
        other_player = state.other.sign
        state_list = []
        for mini_board in state.board_array:
            for row in mini_board:
                for cell in row:
                    if cell == curr_player:
                        state_list.append(1)
                    elif cell == other_player:
                        state_list.append(-1)
                    else:
                        state_list.append(0)
        return torch.tensor(state_list, dtype=torch.float)

    def save_play(self, state: torch.tensor, action: torch.tensor, reward: int, next_state: torch.tensor, game_over: bool) -> None:
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_momory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = self.memory.sample(BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    
    def train_short_memory(self, state: torch.tensor, action: torch.tensor, reward: int, next_state: torch.tensor, game_over: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state: torch.tensor) -> torch.tensor:
        pred_action = self.model(state)
        action = torch.zeros(81)
        legal_actions = torch.nonzero(pred_action)
        self.epsilon = 80 - self.num_games
        if random.randint(0, 200) < self.epsilon:
            rand_index = random.randint(legal_actions.shape[0])
            action[rand_index] = 1
        else:
            action[torch.argmax(pred_action)] = 1
        return action
        
    def score_action(self, action_tensor: torch.tensor, state: "UtttState") -> List[int, bool, "UtttState", str]:
        score_curr_state, _, _ = self.score_state(state)
        next_state = result(state, self.action_tensor2action(action_tensor))
        score_next_state, game_over, winner = self.score_state(next_state)
        return score_next_state - score_curr_state, game_over, next_state, winner

    def score_state(self, state: "UtttState") -> List[int, bool, str]:
        game_over, winner = terminal_test(state)
        score = 0
        player = self.sign
        if player == "X":
            other_player = "O"
        else:
            other_player = "X"
        if game_over:
            if winner == player:
                return 100
            elif winner == other_player:
                return -100
            else: ## TIE
                return -20

        for mini_board in state.board_array:
            mini_board_over, mini_board_winner = terminal_test_mini(mini_board)
            if mini_board_over:
                if mini_board_winner == player:
                    score += 10
                elif mini_board_winner == other_player:
                    score -= 10
        return score, game_over, winner

    def action_tensor2action(self, action_tensor: torch.tensor) -> Tuple[int, int, int]:
        action = torch.argmax(action_tensor)
        mini_board = action // 9
        row = (action % 9) // 3
        col = (action % 9) % 3
        return (mini_board, row, col)


def train():
    winners = []
    rl_agent = RLAgent()
    alpha_beta_player = AlphaBetaPlayer("X")
    game_state = UtttState(rl_agent, alpha_beta_player)

    for _ in range(1):
        state_tensor = rl_agent.get_state(game_state)

        action = rl_agent.get_action(state_tensor)

        reward, game_over, game_state, winner = rl_agent.score_action(action, game_state)

        rl_agent.train_short_memory(state_tensor, action, reward, rl_agent.get_state(game_state), game_over)

        rl_agent.save_play(state_tensor, action, reward, rl_agent.get_state(game_state), game_over)

        ## Make the move for the alpha beta player
        action_alpha_beta = alpha_beta_player.make_move(game_state)
        game_state = result(game_state, action_alpha_beta)
        game_over_alpha_beta, winner_alpha_beta = terminal_test(game_state)

        if game_over:
            game_state = UtttState(rl_agent, alpha_beta_player)
            rl_agent.num_games += 1
            rl_agent.train_long_momory()
            winners.append("Rl")
        elif game_over_alpha_beta:
            game_state = UtttState(rl_agent, alpha_beta_player)
            rl_agent.num_games += 1
            rl_agent.train_long_momory()
            winners.append("Alpha Beta")

    return winners

from gameuttt import UtttState, terminal_test, terminal_test_mini, result
from players import RandomPlayer, HumanPlayer, MinimaxPlayer, AlphaBetaPlayer

if __name__ == "__main__":
    random_player = RandomPlayer("X")
    human_player = HumanPlayer("O")
    uttt_state = UtttState(random_player, human_player)
    print("Uttt state \n",uttt_state)
    agent = RLAgent()
    torch_state = agent.get_state(uttt_state)
    torch_state[5] = 100
    torch_state[8] = 100
    torch_state[9] = 100
    non_zero = torch.nonzero(torch_state)
    print(non_zero)
    print(non_zero.shape)