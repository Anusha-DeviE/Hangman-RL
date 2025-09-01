import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # Simple MLP for feature extraction
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.net(observations)

class LSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(LSTMPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        # Add an LSTM layer
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.lstm_hidden = None
    
    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)
        # Add batch dimension for lstm: (batch, seq_len=1, features)
        features = features.unsqueeze(1)
        if self.lstm_hidden is None or features.size(0) != self.lstm_hidden[0].size(1):
            # Initialize hidden/cell states
            h0 = torch.zeros(1, features.size(0), 128).to(features.device)
            c0 = torch.zeros(1, features.size(0), 128).to(features.device)
            self.lstm_hidden = (h0, c0)
        
        lstm_out, self.lstm_hidden = self.lstm(features, self.lstm_hidden)
        lstm_out = lstm_out.squeeze(1)
        
        distribution = self._get_action_dist_from_latent(lstm_out, lstm_out)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(lstm_out)
        return actions, values, log_prob
    
    def predict_values(self, obs):
        features = self.features_extractor(obs)
        lstm_out, _ = self.lstm(features.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)
        return self.value_net(lstm_out)

def create_ppo_lstm_agent(env):
    model = PPO(policy=LSTMPolicy, env=env, verbose=1, n_steps=256, batch_size=128,
                learning_rate=1e-4, gamma=0.99, ent_coef=0.01)
    return model
