import argparse
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes()

from baselines.bench.monitor import load_results

matplotlib.rcParams.update({'font.size': 30})

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func

def ts2xy(ts, xaxis):
    if xaxis == X_TIMESTEPS:
        x = np.cumsum(ts.l.values)
        y = ts.r.values
    elif xaxis == X_EPISODES:
        x = np.arange(len(ts))
        y = ts.r.values
    elif xaxis == X_WALLTIME:
        x = ts.t.values / 3600.
        y = ts.r.values
    else:
        raise NotImplementedError
    return x, y


def plot_results_single(ax, input_dir, num_timesteps, xaxis):
    ts = load_results(input_dir)
    ts = ts[ts.l.cumsum() <= num_timesteps]
    xy_list = ts2xy(ts, xaxis)

    x = xy_list[0]
    y = xy_list[1]
    ax.plot(x, y, alpha=0.4, linewidth=0.8, c=sns.color_palette()[0])
    x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
    print ("avg_100: %.1f" % np.mean(y_mean[-100:]))
    ax.plot(x, y_mean, linewidth=0.8, c=sns.color_palette()[0], label='normal')

    # plt.set_title(title)
    # ax.set_ylabel("Episode Rewards")
    # ax.legend()
    # plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='Path of log directory', default='logs')
    parser.add_argument('--num_timesteps', type=int, default=int(5e7))
    parser.add_argument('--xaxis', help='Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--task_name', help='Name of atari game', default='Pong')
    parser.add_argument('--save_dir', help = 'Directory of output plots', default='../results')
    parser.add_argument('--noise_type', type=str, help='noise type (norm_one/norm_all/anti_iden)',
                        default='anti_iden')
    parser.add_argument('--plot_normal', type=str, help='whether to plot baseline with normal rewards')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, "paper")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dirs = glob.glob(os.path.join(args.log_dir, "openai*"))
    dirs = sorted(dirs)

    for input_dir in dirs:

        with open(os.path.join(input_dir, "setting.txt"), "r") as f:
            line = f.readlines()[-1].rstrip()
            # normal = line.split()[1][0:-1].split(',')[0]
            weight = float(line.split()[3][0:-1].split(',')[0])
            surrogate = line.split()[5][0:-1].split(',')[0]
            # noise_type = line.split()[7][0:-1].split(')')[0]
            if weight in [0.1, 0.3, 0.7, 0.9] and surrogate == 'True':
                print ("-" * 20)
                print (line)
                plot_results_single(plt, input_dir, args.num_timesteps, args.xaxis)
                print ("-" * 20)

if __name__ == '__main__':
    main()
