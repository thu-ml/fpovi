from experiments.master import runner
import os
import logging
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-max_gpus', default='7', type=eval)
parser.add_argument('-n_multiplex', default=2, type=int)
parser.add_argument('-bandit', default='mushroom', type=str)
args = parser.parse_args()


logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

slave_dir = os.path.abspath('.')
root_cmd = 'python main.py --layers 100,100 --bandit {}'.format(args.bandit)
param_specs = {
    '-num_context': [40000],
    '-seed': list(range(1334, 1344))
}

import datetime
dt = datetime.datetime.now()
timestr = '{}-{}-{}-{}'.format(dt.month, dt.day, dt.hour, dt.minute)
log_dir = os.path.join(slave_dir, '../../run/bandit-{}-{}/'.format(args.bandit, timestr))
os.makedirs(os.path.join(log_dir, "test_{}".format(dt.second))) # make sure we can write

tasks = runner.list_tasks(
    root_cmd,
    param_specs,
    slave_dir,
    log_dir)

for i, t in enumerate(tasks):
    ncmd = t.cmd.replace('-dir', '--logdir').replace('-production ', '')
    tasks[i] = t._replace(cmd=ncmd)

print('\n'.join([t.cmd for t in tasks]))
print(args)

# r = runner.Runner(n_max_gpus=[0,3], n_multiplex=5, n_max_retry=-1)
# r = runner.Runner(n_max_gpus=7, n_multiplex=2, n_max_retry=-1)
r = runner.Runner(n_max_gpus=args.max_gpus, n_multiplex=args.n_multiplex, n_max_retry=-1)
r.run_tasks(tasks)
