import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim=41, output_dim=26):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.model = DQN()
        self.target_model = DQN()
        self.update_target()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 25)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
