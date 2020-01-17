import argparse
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_color_codes()

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default="logs/dqn_cartpole",
                    help='The path of log directory [default: logs/dqn_cartpole')
parser.add_argument('--all', type=bool, default=False,
                    help='Plot all the curves (diff errs) [default: False]')
parser.add_argument('--weight', type=float, default=0.2,
                    help='Weight of noise [default: 0.2]')
FLAGS = parser.parse_args()
LOG_DIR = FLAGS.log_dir
WEIGHT = FLAGS.weight


def smooth(y, weight=0.6):
    last = y[0]
    smoothed = []
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def plot_qlearn_cartpole_all():
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))['0']
    plt.plot(smooth(list(history_normal)), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    cnt = 0
    for err in [0.2, 0.4, 0.6, 0.8]:
        history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "noisy.csv"))['0']
        history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "surrogate.csv"))['0']
        plt.plot(smooth(list(history_noisy)), linewidth=1.5, c=sns.color_palette()[cnt+1], label="noisy (" + str(err) + ")")
        plt.plot(list(history_noisy), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+1])
        plt.plot(smooth(list(history_surrogate)), linewidth=1.5, c=sns.color_palette()[cnt+2], label="surrogate (" + str(err) + ")")
        plt.plot(list(history_surrogate), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+2])
        cnt += 2

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps)')
    plt.legend(loc='best')
    plt.savefig(os.path.join(LOG_DIR, "CartPole-v0-reward-all (Q-Learning).png"))


def plot_qlearn_cartpole(weight=0.2):
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))['0']
    history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "noisy.csv"))['0']
    history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "surrogate.csv"))['0']

    plt.plot(smooth(list(history_normal)), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy)), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate)), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='best')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-steps-" + str(weight) + " (Q-Learning).png"))


def plot_dqn_cartpole_all():
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    cnt = 0
    for err in [0.2, 0.4, 0.5]:
        history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "noisy.csv"))
        history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "surrogate.csv"))
        plt.plot(smooth(list(history_noisy['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[cnt+1], label="noisy (" + str(err) + ")")
        plt.plot(list(history_noisy['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+1])
        plt.plot(smooth(list(history_surrogate['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[cnt+2], label="surrogate (" + str(err) + ")")
        plt.plot(list(history_surrogate['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+2])
        cnt += 2

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps)')
    plt.legend(loc='best')
    plt.savefig(os.path.join(LOG_DIR, "CartPole-v0-reward-all (DQN).png"))


def plot_dqn_cartpole(weight=0.2):
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "noisy.csv"))
    history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "surrogate.csv"))

    plt.plot(smooth(list(history_normal['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='best')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-steps-" + str(weight) + " (DQN).png"))

    plt.clf()
    plt.plot(smooth(list(history_normal['episode_reward'])), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy['episode_reward'])), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate['episode_reward'])), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])
    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (reward-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-reward-" + str(weight) + " (DQN).png"))


def plot_sarsa_cartpole_all():
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    cnt = 0
    for err in [0.2, 0.4, 0.5]:
        history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "noisy.csv"))
        history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "surrogate.csv"))
        plt.plot(smooth(list(history_noisy['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[cnt+1], label="noisy (" + str(err) + ")")
        plt.plot(list(history_noisy['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+1])
        plt.plot(smooth(list(history_surrogate['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[cnt+2], label="surrogate (" + str(err) + ")")
        plt.plot(list(history_surrogate['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+2])
        cnt += 2

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps)')
    plt.legend(loc='best')
    plt.savefig(os.path.join(LOG_DIR, "CartPole-v0-steps-all (SARSA).png"))


def plot_sarsa_cartpole(weight=0.2):
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "noisy.csv"))
    history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "surrogate.csv"))

    plt.plot(smooth(list(history_normal['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='best')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-steps-" + str(weight) + " (SARSA).png"))

    plt.clf()
    plt.plot(smooth(list(history_normal['episode_reward'])), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy['episode_reward'])), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate['episode_reward'])), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])
    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (reward-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-reward-" + str(weight) +  " (SARSA).png"))


def plot_cem_cartpole_all():
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    cnt = 0
    for err in [0.2, 0.4, 0.5]:
        history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "noisy.csv"))
        history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(err)), "surrogate.csv"))
        plt.plot(smooth(list(history_noisy['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[cnt+1], label="noisy (" + str(err) + ")")
        plt.plot(list(history_noisy['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+1])
        plt.plot(smooth(list(history_surrogate['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[cnt+2], label="surrogate (" + str(err) + ")")
        plt.plot(list(history_surrogate['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+2])
        cnt += 2

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps)')
    plt.legend(loc='best')
    plt.savefig(os.path.join(LOG_DIR, "CartPole-v0-reward-all (CEM).png"))


def plot_cem_cartpole(weight=0.2):
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    history_noisy = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "noisy.csv"))
    history_surrogate = pandas.read_csv(os.path.join(os.path.join(LOG_DIR, str(weight)), "surrogate.csv"))

    plt.plot(smooth(list(history_normal['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate['nb_episode_steps'])), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate['nb_episode_steps']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])

    plt.ylabel('steps per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (steps-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='best')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-steps-" + str(weight) + " (CEM).png"))

    plt.clf()
    plt.plot(smooth(list(history_normal['episode_reward'])), linewidth=1.5, c=sns.color_palette()[0])
    plt.plot(smooth(list(history_noisy['episode_reward'])), linewidth=1.5, c=sns.color_palette()[1])
    plt.plot(smooth(list(history_surrogate['episode_reward'])), linewidth=1.5, c=sns.color_palette()[2])
    plt.plot(list(history_normal['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    plt.plot(list(history_noisy['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(list(history_surrogate['episode_reward']), alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])
    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('CartPole-v0 (reward-' + str(weight) + ")")
    plt.legend(['normal', 'noisy', 'surrogate'], loc='upper right')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "CartPole-v0-reward-" + str(weight) +  " (CEM).png"))


def plot_ddpg_pendulum_all():
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['episode_reward'] / 200.0)), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['episode_reward'] / 200.0), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    cnt = 0
    for err in [0.2, 0.4, 0.5]:
        reward_noisy = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(err)), "noisy_reward")))
        reward_surrogate = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(err)), "surrogate_reward")))
        plt.plot(smooth(reward_noisy), linewidth=1.5, c=sns.color_palette()[cnt+1], label="noisy (" + str(err) + ")")
        plt.plot(reward_noisy, alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+1])
        plt.plot(smooth(reward_surrogate), linewidth=1.5, c=sns.color_palette()[cnt+2], label="surrogate (" + str(err) + ")")
        plt.plot(reward_surrogate, alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+2])
        cnt += 2

    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('Pendulum-v0 (reward)')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(LOG_DIR, "Pendulum-v0-reward-all (DDPG).png"))


def plot_ddpg_pendulum(weight=0.2):
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['episode_reward'] / 200.0)), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['episode_reward'] / 200.0), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    reward_noisy = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(weight)), "noisy_reward")))
    reward_surrogate = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(weight)), "surrogate_reward")))
    plt.plot(smooth(reward_noisy), linewidth=1.5, c=sns.color_palette()[1], label="noisy")
    plt.plot(reward_noisy, alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(smooth(reward_surrogate), linewidth=1.5, c=sns.color_palette()[2], label="surrogate")
    plt.plot(reward_surrogate, alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])

    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('Pendulum-v0 (reward-' + str(weight) + ")")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "Pendulum-v0-reward-" + str(weight) + " (DDPG).png"))


def plot_naf_pendulum_all():
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['episode_reward'] / 2.0)), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['episode_reward'] / 2.0), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    cnt = 0
    for err in [0.2, 0.4, 0.5]:
        reward_noisy = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(err)), "noisy_reward")))
        reward_surrogate = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(err)), "surrogate_reward")))
        plt.plot(smooth(reward_noisy), linewidth=1.5, c=sns.color_palette()[cnt+1], label="noisy (" + str(err) + ")")
        plt.plot(reward_noisy, alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+1])
        plt.plot(smooth(reward_surrogate), linewidth=1.5, c=sns.color_palette()[cnt+2], label="surrogate (" + str(err) + ")")
        plt.plot(reward_surrogate, alpha=0.4, linewidth=0.8, c=sns.color_palette()[cnt+2])
        cnt += 2

    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('Pendulum-v0 (reward)')
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(LOG_DIR, "Pendulum-v0-reward-all (NAF).png"))


def plot_naf_pendulum(weight=0.2):
    history_normal = pandas.read_csv(os.path.join(LOG_DIR, "normal.csv"))
    plt.plot(smooth(list(history_normal['episode_reward'] / 2.0)), linewidth=1.5, c=sns.color_palette()[0], label="normal")
    plt.plot(list(history_normal['episode_reward'] / 2.0), alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])

    reward_noisy = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(weight)), "noisy_reward")))
    reward_surrogate = list(np.loadtxt(os.path.join(os.path.join(LOG_DIR, str(weight)), "surrogate_reward")))
    plt.plot(smooth(reward_noisy), linewidth=1.5, c=sns.color_palette()[1], label="noisy")
    plt.plot(reward_noisy, alpha=0.4, linewidth=0.8, c=sns.color_palette()[1])
    plt.plot(smooth(reward_surrogate), linewidth=1.5, c=sns.color_palette()[2], label="surrogate")
    plt.plot(reward_surrogate, alpha=0.4, linewidth=0.8, c=sns.color_palette()[2])

    plt.ylabel('reward per episode')
    plt.xlabel('episode')
    plt.title('Pendulum-v0 (reward-' + str(weight) + ")")
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(os.path.join(os.path.join(LOG_DIR, str(weight)), "Pendulum-v0-reward-" + str(weight) + " (NAF).png"))


def plot():
    if "qlearn" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_qlearn_cartpole(weight=WEIGHT)
    elif "dqn" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_dqn_cartpole(weight=WEIGHT)
    elif "sarsa" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_sarsa_cartpole(weight=WEIGHT)
    elif "cem" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_cem_cartpole(weight=WEIGHT)
    elif "ddpg" in LOG_DIR and "pendulum" in LOG_DIR:
        plot_ddpg_pendulum(weight=WEIGHT)
    elif "naf" in LOG_DIR and "pendulum" in LOG_DIR:
        plot_naf_pendulum(weight=WEIGHT)
    else:
        raise NotImplementedError


def plot_all():
    if "qlearn" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_qlearn_cartpole_all()
    elif "dqn" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_dqn_cartpole_all()
    elif "sarsa" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_sarsa_cartpole_all()
    elif "cem" in LOG_DIR and "cartpole" in LOG_DIR:
        plot_cem_cartpole_all()
    elif "ddpg" in LOG_DIR and "pendulum" in LOG_DIR:
        plot_ddpg_pendulum_all()
    elif "naf" in LOG_DIR and "pendulum" in LOG_DIR:
        plot_naf_pendulum_all()
    else:
        raise NotImplementedError



if __name__ == "__main__":
    if FLAGS.all:
        plot_all()
    else:
        plot()
