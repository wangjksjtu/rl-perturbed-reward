import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from baselines.bench.monitor import load_results

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

def plot_curves(xy_list, xaxis, title):
    plt.figure(figsize=(8,2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean) #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def plot_results(dirs, num_timesteps, xaxis, task_name):
    tslist = []
    for dir in dirs:
        ts = load_results(dir)
        ts = ts[ts.l.cumsum() <= num_timesteps]
        tslist.append(ts)
    xy_list = [ts2xy(ts, xaxis) for ts in tslist]
    plot_curves(xy_list, xaxis, task_name)

# Example usage in jupyter-notebook
# from baselines import log_viewer
# %matplotlib inline
# log_viewer.plot_results(["./log"], 10e6, log_viewer.X_TIMESTEPS, "Breakout")
# Here ./log is a directory containing the monitor.csv files

def main():
    import argparse
    import os
    import glob
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='Path of log directory', default='logs')
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help = 'Varible on X-axis', default = X_TIMESTEPS)
    parser.add_argument('--task_name', help = 'Title of plot', default = 'PongNoFrameskip-v4')
    parser.add_argument('--weight', help = 'Weight of noise', default = 0.2, type=float)
    parser.add_argument('--save_dir', help = 'Didrectory of output plots', default = 'results')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dirs = glob.glob(os.path.join(args.log_dir, "*"))
    sorted(dirs)
    cnt = 0
    for directory in dirs:
        print directory
        with open(os.path.join(directory, "setting.txt"), "r") as f:
            line = f.readlines()[-1].rstrip()
            print (line.split())
            normal = line.split()[1][0:-1].split(',')[0]
            weight = float(line.split()[3][0:-1].split(',')[0])
            surrogate = line.split()[5][0:-1].split(',')[0]
            noise_type = line.split()[7][0:-1].split(')')[0]
            print (normal, weight, surrogate, noise_type)
        if normal == 'True':
            title = args.task_name + " (normal)"
        elif surrogate == 'False':
            title = args.task_name + " (noisy-" + str(weight) + "-" + noise_type + ")"
        else:
            title = args.task_name + " (surrogate-" + str(weight) + "-" + noise_type + ")"
        
        print (weight, args.weight)
        if weight == args.weight:
            print (args.weight)
            plot_results([directory], args.num_timesteps, args.xaxis, title)
            plt.savefig(os.path.join(args.save_dir, title + ".png"))
        cnt += 1
    print cnt


if __name__ == '__main__':
    main()
