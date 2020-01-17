import argparse
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='logs/ddpg_pendulum/norm_one',
                    help='Log dir [default: logs/ddpg_pendulum/norm_one]')
parser.add_argument('--save_dir', default='docs/ddpg_pendulum/norm_one',
                    help='Path of directory to saved [default: docs/ddpg_pendulum/norm_one]')
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
SAVE_DIR = FLAGS.save_dir

assert (os.path.exists(LOG_DIR))
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def collect():
    for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        input_dir = os.path.join(LOG_DIR, str(j))
        files = glob.glob(os.path.join(input_dir, "*.png"))
        for fin in files:
            filename = fin[fin.rindex("/")+1:]
            fout = os.path.join(SAVE_DIR, filename)
            print "cp '%s' '%s'" % (fin, fout)
            os.system("cp '%s' '%s'" % (fin, fout))


if __name__ == "__main__":
    collect()
