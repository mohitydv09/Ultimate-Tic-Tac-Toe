import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): ## 81, 1024, 1024, 81
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def forward(self, input):
        output = F.relu(self.linear1(input))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output

    def init_weights(self):
        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear3.weight, nonlinearity='linear')

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        if isinstance(state, tuple):
            state =         torch.tensor(np.array(state), dtype=torch.float)
            next_state =    torch.tensor(np.array(next_state), dtype=torch.float)
            action =        torch.tensor(np.array(action), dtype=torch.long)
            reward =        torch.tensor(np.array(reward), dtype=torch.float)
        else:
            state =         torch.tensor(state, dtype=torch.float)
            next_state =    torch.tensor(next_state, dtype=torch.float)
            action =        torch.tensor(action, dtype=torch.long)
            reward =        torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        # print("Trainer Ran, Loss was: ", loss.item())