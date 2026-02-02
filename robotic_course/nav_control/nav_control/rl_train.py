import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPG
from rl_env import PathEnv
import os
import torch


def random_path(n=4, size=3.0):
    path = [(0.0, 0.0)]
    for _ in range(n - 1):
        lx, ly = path[-1]
        dx = np.random.uniform(0.5, 1.0)
        dy = np.random.uniform(-0.3, 0.3)
        nx = np.clip(lx + dx, 0, size)
        ny = np.clip(ly + dy, -size/2, size/2)
        path.append((nx, ny))
    return path


def make_paths(n=100):
    paths = []
    for _ in range(n):
        wp = np.random.randint(3, 6)
        paths.append(random_path(wp))
    return paths


def train(episodes=1000, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    env = PathEnv()
    agent = DDPG(
        state_dim=env.observation_shape,
        action_dim=env.num_actions,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        noise_sigma=0.15
    )

    paths = make_paths(50)
    rewards = []
    successes = []

    for ep in range(episodes):
        path = paths[ep % len(paths)]
        env.set_path(path)

        sx, sy = path[0]
        syaw = np.random.uniform(-0.2, 0.2)
        env.set_robot_pose(sx, sy, syaw)

        state, _ = env.reset()
        agent.noise.reset()
        total = 0
        ok = False

        for _ in range(env.max_steps):
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, trunc, _ = env.step(
                torch.tensor([action], dtype=torch.float32)
            )

            agent.store_transition(state, action, reward, next_state, float(done or trunc))
            agent.optimize()

            state = next_state
            total += reward

            if done or trunc:
                if env.wp_idx >= len(path):
                    ok = True
                break

        rewards.append(total)
        successes.append(1 if ok else 0)

        if len(successes) > 100:
            successes.pop(0)

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            rate = np.mean(successes) * 100
            print(f'Episode {ep+1}/{episodes} | Reward: {avg:.1f} | Success: {rate:.0f}%')

        if (ep + 1) % 200 == 0:
            agent.save(os.path.join(save_dir, f'ddpg_ep{ep+1}.pth'))

    agent.save(os.path.join(save_dir, 'ddpg_final.pth'))
    plot(rewards, save_dir)
    return agent


def plot(rewards, save_dir):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)
    if len(rewards) >= 50:
        avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), avg, 'r', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=50)
    plt.xlabel('Reward')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.close()


def evaluate(agent, n=10):
    env = PathEnv()
    ok = 0

    for i in range(n):
        path = random_path(4)
        env.set_path(path)
        env.set_robot_pose(0, 0, 0)

        state, _ = env.reset()
        total = 0

        for _ in range(env.max_steps):
            action = agent.select_action(state, add_noise=False)
            state, reward, done, trunc, _ = env.step(
                torch.tensor([action], dtype=torch.float32)
            )
            total += reward
            if done or trunc:
                if env.wp_idx >= len(path):
                    ok += 1
                break

        print(f'Eval {i+1}: {total:.1f}')

    print(f'Success: {ok}/{n}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model', type=str, default='models/ddpg_final.pth')
    args = parser.parse_args()

    if args.evaluate:
        env = PathEnv()
        agent = DDPG(state_dim=env.observation_shape, action_dim=env.num_actions)
        agent.load(args.model)
        evaluate(agent)
    else:
        agent = train(episodes=args.episodes, save_dir=args.save_dir)
        evaluate(agent)
