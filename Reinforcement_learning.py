
# Inverted pendulum problem using reinforcement learning.

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Memory buffer
class MemoryBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, 3), dtype=np.float32)
        self.action = np.zeros((max_size, 1), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, 3), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def push(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.done[idx])

# Actor network
def get_actor(action_bound):
    inputs = layers.Input(shape=(3,))
    out = layers.Dense(256, activation='relu')(inputs)
    out = layers.Dense(256, activation='relu')(out)
    outputs = layers.Dense(1, activation='tanh')(out)
    outputs = outputs * action_bound
    return tf.keras.Model(inputs, outputs)

# Critic network
def get_critic():
    state_input = layers.Input(shape=(3,))
    action_input = layers.Input(shape=(1,))
    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(256, activation='relu')(concat)
    out = layers.Dense(256, activation='relu')(out)
    outputs = layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs)

# Twin Delayed Deep Deterministic (TD3) Policy Gradient python class
class TD3Agent:
    def __init__(self, action_bound):
        self.action_bound = action_bound
        self.actor = get_actor(action_bound)
        self.actor_target = get_actor(action_bound)
        self.actor_target.set_weights(self.actor.get_weights())

        self.critic1 = get_critic()
        self.critic1_target = get_critic()
        self.critic1_target.set_weights(self.critic1.get_weights())

        self.critic2 = get_critic()
        self.critic2_target = get_critic()
        self.critic2_target.set_weights(self.critic2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0

    def get_action(self, state, noise_scale):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        noise = noise_scale * np.random.randn(*action.shape)
        action = action + noise
        return np.clip(action, -self.action_bound, self.action_bound)

    def train(self, memory_buffer, batch_size=100):
        if memory_buffer.size < batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = memory_buffer.sample(batch_size)
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)


        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones)

        # policy settings
        noise = tf.clip_by_value(
            tf.random.normal(shape=(batch_size, 1), stddev=self.policy_noise),
            -self.noise_clip, self.noise_clip
        )
        next_actions = self.actor_target(next_states) + noise
        next_actions = tf.clip_by_value(next_actions, -self.action_bound, self.action_bound)

        # Target Q values
        target_q1 = self.critic1_target([next_states, next_actions])
        target_q2 = self.critic2_target([next_states, next_actions])
        target_q = tf.minimum(target_q1, target_q2)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        # Update critics
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            current_q1 = self.critic1([states, actions])
            current_q2 = self.critic2([states, actions])
            critic1_loss = tf.reduce_mean((current_q1 - target_q) ** 2)
            critic2_loss = tf.reduce_mean((current_q2 - target_q) ** 2)

        critic1_grads = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actions_pred = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic1([states, actions_pred]))

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Soft update target networks
            self._update_target(self.actor.variables, self.actor_target.variables)
            self._update_target(self.critic1.variables, self.critic1_target.variables)
            self._update_target(self.critic2.variables, self.critic2_target.variables)

    def _update_target(self, vars, target_vars):
        for var, target_var in zip(vars, target_vars):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)

def main():
    env = gym.make("Pendulum-v1")
    tf.random.set_seed(0)
    np.random.seed(0)

    max_action = float(env.action_space.high[0])
    agent = TD3Agent(max_action)
    memory_buffer = MemoryBuffer()

    Iterations = 100
    batch_size = 100
    iteration_rewards = []

    for it in range(Iterations):
        state, _ = env.reset()
        it_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, noise_scale=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            it_reward += reward

            agent.train(memory_buffer, batch_size)

        iteration_rewards.append(it_reward)
        print(f"Iteration {it + 1}: Reward = {it_reward:.2f}")

    # Plot results
    plt.plot(iteration_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Total Reward')
    plt.title('TD3 on Pendulum-v1')
    plt.show()

if __name__ == "__main__":
    main()