# Reinforcement-Learning

# Inverted Pendulum Swing-Up with TD3 (Reinforcement Learning)

This project implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm to solve the Pendulum-v1 environment — a classic inverted pendulum swing-up problem using reinforcement learning.

# Problem Statement

The inverted pendulum swing-up problem is a classical control task where:

-A pendulum starts at a random angle.

-The agent’s objective is to apply torque to swing the pendulum upright and keep it balanced.

-The agent receives a continuous reward signal based on how well it maintains the upright position with minimal effort.

# Algorithm: TD3

This project uses the TD3 algorithm, an improvement over DDPG that:

-Uses twin critics to reduce overestimation bias.

-Adds target policy smoothing.

-Implements delayed actor updates for stability.
