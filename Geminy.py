import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pettingzoo.classic import tictactoe_v3

# --- Hyperparamètres ---
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 500
LEARNING_RATE = 1e-4


# --- Le Modèle (Réseau de Neurones) ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# --- L'Agent ---
class Agent:
    def __init__(self, env):
        # Observation: 3x3x2 (plat -> 18), Actions: 9
        self.input_dim = 18
        self.action_dim = 9

        self.online_net = DQN(self.input_dim, self.action_dim)
        self.target_net = DQN(self.input_dim, self.action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.step_count = 0

    def preprocess_obs(self, obs):
        # Aplatir l'observation du plateau (3, 3, 2) -> (18,)
        return np.array(obs["observation"]).flatten()

    def choose_action(self, obs_dict, training=True):
        obs = self.preprocess_obs(obs_dict)
        mask = obs_dict["action_mask"]

        # Epsilon-Greedy
        if training and random.random() < self.epsilon:
            available_actions = [i for i, m in enumerate(mask) if m == 1]
            return random.choice(available_actions)

        # Prédiction du réseau
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(obs_t)

        # Action Masking : On met les actions illégales à -infini
        # pour s'assurer qu'elles ne soient jamais choisies par argmax
        q_values_np = q_values.numpy().flatten()
        q_values_np[mask == 0] = -1e9

        return np.argmax(q_values_np)

    def store_transition(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.replay_buffer.append(
            (
                self.preprocess_obs(obs),
                action,
                reward,
                self.preprocess_obs(next_obs),
                done,
                mask,
                next_mask,
            )
        )

    def train_step(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        obses, actions, rewards, next_obses, dones, masks, next_masks = zip(*batch)

        obses_t = torch.FloatTensor(np.array(obses))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_obses_t = torch.FloatTensor(np.array(next_obses))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        next_masks_t = torch.IntTensor(np.array(next_masks))

        # Calcul Q(s, a)
        current_q_values = self.online_net(obses_t).gather(1, actions_t)

        # Calcul Max Q(s', a') avec Target Net et Action Masking
        with torch.no_grad():
            next_q_values = self.target_net(next_obses_t)
            # Appliquer le masque sur le prochain état (très important pour DQN)
            # On remplace les actions impossibles par une très petite valeur
            min_val = torch.tensor(-1e9)
            next_q_values = torch.where(next_masks_t == 1, next_q_values, min_val)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + (1 - dones_t) * GAMMA * max_next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Epsilon
        self.epsilon = max(
            EPSILON_END, EPSILON_START - (self.step_count / EPSILON_DECAY)
        )
        self.step_count += 1

        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())


# --- Entraînement ---
def train_agent():
    env = tictactoe_v3.env()
    agent = Agent(env)

    print("Début de l'entraînement (environ 1-2 minutes)...")

    # Dictionnaire pour stocker l'état précédent de chaque agent afin de calculer la transition
    # Car PettingZoo donne la récompense au tour SUIVANT
    last_obs = {}
    last_action = {}

    # Nombre d'épisodes (parties)
    n_games = 5000

    for game in range(n_games):
        env.reset()
        if game % 500 == 0:
            print(f"Partie {game}/{n_games} - Epsilon: {agent.epsilon:.2f}")

        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            # Stocker la transition pour l'agent qui a joué au tour PRÉCÉDENT
            # Si 'player_1' joue maintenant, 'player_2' reçoit sa récompense/état final
            if agent_name in last_obs:
                prev_obs_dict = last_obs[agent_name]
                prev_act = last_action[agent_name]

                agent.store_transition(
                    prev_obs_dict,
                    prev_act,
                    reward,
                    observation,  # Next state
                    done,
                    prev_obs_dict["action_mask"],
                    observation["action_mask"],
                )
                agent.train_step()

            if not done:
                action = agent.choose_action(observation, training=True)
                last_obs[agent_name] = observation
                last_action[agent_name] = action
                env.step(action)
            else:
                env.step(None)
                # Nettoyer la mémoire pour la prochaine partie
                last_obs = {}
                last_action = {}

    print("Entraînement terminé.")
    torch.save(agent.online_net.state_dict(), "tictactoe_dqn.pth")
    return agent


# --- Jeu vs Humain ---
def play_vs_human(agent=None):
    if agent is None:
        env = tictactoe_v3.env()
        agent = Agent(env)
        try:
            agent.online_net.load_state_dict(torch.load("tictactoe_dqn.pth"))
            print("Modèle chargé.")
        except:
            print("Aucun modèle trouvé, l'IA jouera au hasard.")

    env = tictactoe_v3.env(render_mode="human")
    env.reset()

    print("\n--- Début du match Humain vs AI ---")
    print("Vous jouez les 'player_1' (X).")

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if reward == 1:
                print(f"\n{agent_name} a gagné !")
            elif reward == -1:  # L'autre joueur a gagné au tour d'avant
                pass
            else:
                print("\nMatch nul !")
            env.step(None)
            continue

        if agent_name == "player_1":
            # Tour de l'humain
            mask = observation["action_mask"]
            valid_actions = [i for i, m in enumerate(mask) if m == 1]
            print(f"\nCases disponibles : {valid_actions}")

            action = -1
            while action not in valid_actions:
                try:
                    action = int(input("Votre coup (0-8) : "))
                except ValueError:
                    pass
            env.step(action)

        else:
            # Tour de l'IA
            action = agent.choose_action(observation, training=False)
            print(f"L'IA joue en : {action}")
            env.step(action)


if __name__ == "__main__":
    # 1. Entraîner le modèle
    trained_agent = train_agent()

    # 2. Jouer contre le modèle
    while True:
        play_vs_human(trained_agent)
        if input("Rejouer ? (o/n) : ").lower() != "o":
            break
