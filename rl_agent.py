import torch
from collections import deque
import random
from typing import List, Tuple
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000


class RLAgent:
    def __init__(self, sign = "X"):
        self.num_games = 0
        self.epsilon = 0 ## Exploration rate
        self.gamma = 0.9 ## Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(81, 512, 81)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
        self.sign = sign

    def get_sign(self):
        return self.sign

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
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    
    def train_short_memory(self, state: torch.tensor, action: torch.tensor, reward: int, next_state: torch.tensor, game_over: bool) -> None:
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state: torch.tensor, legal_action: torch.tensor) -> torch.tensor:
        pred_action = self.model(state)
        pred_action_masked = pred_action.masked_fill(legal_action == 0, float('-inf'))

        action_action = torch.zeros(81)
        self.epsilon = 80 - self.num_games
        if random.randint(0, 200) < self.epsilon:
            legal_actions = torch.where(legal_action == 1)[0]
            rand_index = random.randint(0,legal_actions.shape[0]-1)
            action_action[legal_actions[rand_index]] = 1
        else:
            action_action[torch.argmax(pred_action_masked)] = 1
        return action_action
        
    def score_action(self, action_tensor: torch.tensor, state: "UtttState") -> Tuple[int, bool, "UtttState", str]:
        score_curr_state, _, _ = self.score_state(state)
        next_state = result(state, self.action_tensor2action(action_tensor))
        score_next_state, game_over, winner = self.score_state(next_state)
        return score_next_state - score_curr_state, game_over, next_state, winner

    def score_state(self, state: "UtttState") -> Tuple[int, bool, str]:
        game_over, winner = terminal_test(state)
        score = 0
        player = self.sign
        if player == "X":
            other_player = "O"
        else:
            other_player = "X"

        if game_over:
            if winner == player:
                return 100, True, winner
            elif winner == other_player:
                return -100, True, winner
            else: ## TIE
                return -20, True, winner

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

def valid_actions2action_tensor(actions: List[tuple]) -> torch.tensor:
    action_tensor = torch.zeros(81)
    for action in actions:
        mini_board, row, col = action
        action_tensor[mini_board*9 + row*3 + col] = 1
    return action_tensor

def train():
    winners = []
    rl_agent = RLAgent()
    # alpha_beta_player = AlphaBetaPlayer("O", depth_limit=2)
    random_player = RandomPlayer("O")
    game_state = UtttState(rl_agent, random_player)

    for _ in range(50000):
        
        state_tensor = rl_agent.get_state(game_state)

        legal_actions = actions(game_state)

        if len(legal_actions) == 0:
            print("No Legal Actions for RL, match Tie")
            print(game_state)
            game_state = UtttState(rl_agent, random_player)
            rl_agent.num_games += 1
            rl_agent.train_long_momory()
            winners.append("Tie")
            continue

        action_actual_tensor = valid_actions2action_tensor(legal_actions)

        action_tensor = rl_agent.get_action(state_tensor,  action_actual_tensor)

        reward, game_over, game_state, winner = rl_agent.score_action(action_tensor, game_state)

        rl_agent.train_short_memory(state_tensor, action_tensor, reward, rl_agent.get_state(game_state), game_over)

        rl_agent.save_play(state_tensor, action_tensor, reward, rl_agent.get_state(game_state), game_over)

        if game_over:
            print("Game {:6} won by: RL".format(rl_agent.num_games))
            # print(game_state)
            game_state = UtttState(rl_agent, random_player)
            rl_agent.num_games += 1
            rl_agent.train_long_momory()
            winners.append("Rl")
            continue

        # print("Move made by RL: ")
        # print(game_state)
        ## Make the move for the alpha beta player
        action_random = random_player.make_move(game_state)
        if action_random is None:
            print("No Legal Actions for AB, match Tie")
            print(game_state)
            game_state = UtttState(rl_agent, random_player)
            rl_agent.num_games += 1
            rl_agent.train_long_momory()
            winners.append("Tie")
            continue
        game_state = result(game_state, action_random)
        game_over_alpha_beta, winner_alpha_beta = terminal_test(game_state)

        # print("Move made by Alpha Beta: ")
        # print(game_state)

        if game_over_alpha_beta:
            print("Game {:6} won by: Alpha Beta".format(rl_agent.num_games))
            # print(game_state)
            game_state = UtttState(rl_agent, random_player)
            rl_agent.num_games += 1
            rl_agent.train_long_momory()
            winners.append("AB")
            continue

    return winners

from gameuttt import UtttState, terminal_test, terminal_test_mini, result, actions
from players import RandomPlayer, HumanPlayer, MinimaxPlayer, AlphaBetaPlayer

if __name__ == "__main__":
    winners = train()
    print(winners)