"""train file"""

from copy import deepcopy
import numpy as np
import torch

# import torch.nn as nn
from evaluate import evaluate_HIV
from replay_buffer import ReplayBuffer

from writer import Writer
import time

# import random
import torch
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
# from fasthiv import FastHIVPatient as HIVPatient


# from dqn_agent import ProjectAgent
from network import get_network
from writer import Writer

WRITER = Writer()

ENV = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

CONFIG = {
    "nb_neurons": 512,
    "nb_actions": ENV.action_space.n,
    "state_dim": ENV.observation_space.shape[0],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "learning_rate": 0.001,
    "gamma": 0.9,
    "buffer_size": 5_000_000,
    "epsilon_min": 0.01,
    "epsilon_max": 1.0,
    "epsilon_decay_period": 40000,
    "epsilon_delay_decay": 300,
    "batch_size": 2500,
    "gradient_steps": 2,
    "update_target_strategy": "ema",
    "update_target_freq": 50,
    "update_target_tau": 0.005,
    "max_episode": 2_000,
    "criterion": torch.nn.SmoothL1Loss(),
}

MODEL_PATH = "weights/weights_512_244.pth"
MODEL = get_network(
    CONFIG["state_dim"],
    CONFIG["nb_neurons"],
    CONFIG["nb_actions"],
    device=CONFIG["device"],
)

# MODEL.load_state_dict(
#     torch.load(MODEL_PATH, map_location=CONFIG["device"], weights_only=True)
# )


class ProjectAgent:
    """dqn agent class"""

    def __init__(self, verbose=True):
        self.config = CONFIG
        self.verbose = verbose
        self.model = MODEL
        self.device = self.config["device"]
        self.nb_actions = self.config["nb_actions"]
        self.gamma = self.config["gamma"] if "gamma" in self.config.keys() else 0.95
        self.batch_size = (
            self.config["batch_size"] if "batch_size" in self.config.keys() else 100
        )
        buffer_size = (
            self.config["buffer_size"]
            if "buffer_size" in self.config.keys()
            else int(1e5)
        )
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = (
            self.config["epsilon_max"] if "epsilon_max" in self.config.keys() else 1.0
        )
        self.epsilon_min = (
            self.config["epsilon_min"] if "epsilon_min" in self.config.keys() else 0.01
        )
        self.epsilon_stop = (
            self.config["epsilon_decay_period"]
            if "epsilon_decay_period" in self.config.keys()
            else 1000
        )
        self.epsilon_delay = (
            self.config["epsilon_delay_decay"]
            if "epsilon_delay_decay" in self.config.keys()
            else 20
        )
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = (
            self.config["criterion"]
            if "criterion" in self.config.keys()
            else torch.nn.MSELoss()
        )
        lr = (
            self.config["learning_rate"]
            if "learning_rate" in self.config.keys()
            else 0.001
        )
        self.optimizer = (
            self.config["optimizer"]
            if "optimizer" in self.config.keys()
            else torch.optim.Adam(self.model.parameters(), lr=lr)
        )
        self.nb_gradient_steps = (
            self.config["gradient_steps"]
            if "gradient_steps" in self.config.keys()
            else 1
        )
        self.update_target_strategy = (
            self.config["update_target_strategy"]
            if "update_target_strategy" in self.config.keys()
            else "replace"
        )
        self.update_target_freq = (
            self.config["update_target_freq"]
            if "update_target_freq" in self.config.keys()
            else 20
        )
        self.max_episode = (
            self.config["max_episode"] if "max_episode" in self.config.keys() else 50
        )
        self.update_target_tau = (
            self.config["update_target_tau"]
            if "update_target_tau" in self.config.keys()
            else 0.005
        )
        # for training loop
        self.model_counter: int = 0
        self.current_score: float = 0
        if self.verbose:
            WRITER.display(
                f"Agent successfully initalized on device: {self.device}", "sc"
            )

    def gradient_step(self):
        """gradient step"""
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env):
        """train"""
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        self.model_counter: int = 0
        self.current_score: float = 0
        max_reward_t = -1
        while episode < self.max_episode:
            # t0 = time.time()
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == "replace":
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == "ema":
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = (
                        tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                    )
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                WRITER.display(f"Episode: {episode}", "txt")
                WRITER.display(f"Epsilon: {epsilon}", "txt")
                WRITER.display(f"Batch size: {len(self.memory)}", "txt")
                WRITER.display(f"Episode return: {episode_cum_reward}\n", "txt")
                if episode_cum_reward > max_reward_t:
                    max_reward_t = episode_cum_reward
                    torch.save(self.model.state_dict(), f"weights_512_{episode}.pth")

                state, _ = env.reset()
                episode_return.append(episode_cum_reward)

                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

    def greedy_action(self, state):
        """greedy action"""
        with torch.no_grad():
            q = self.model(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(q).item()

    #################################################
    #################################################
    ######## METHODS NEEDED IN THE "main.py" ########
    #################################################
    #################################################

    def act(self, observation, use_random=False):
        """act"""
        return self.greedy_action(observation)

    def load(self):
        """load model"""
        print(MODEL_PATH)
        self.model = get_network(
            self.config["state_dim"],
            self.config["nb_neurons"],
            self.config["nb_actions"],
            device=self.config["device"],
        )
        self.model.load_state_dict(
            torch.load(
                MODEL_PATH, map_location=self.config["device"], weights_only=True
            )
        )
        self.model.eval()


def main():
    """main"""
    WRITER.display("Training", "t")
    WRITER.display("Config", "st")
    WRITER.display(CONFIG, "txt")
    project_agent = ProjectAgent()
    _ = project_agent.train(env=ENV)


if __name__ == "__main__":
    main()
