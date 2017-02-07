from collections import Counter
import babeltrace
import sys
import numpy as np
import pickle
import os.path

save_dir = 'pickle/'

def parse(trace_file, window_size=250):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(save_dir+'windows.p'):
        return pickle.load(open(save_dir+'windows.p', 'rb'))

    # A trace collection contains one or more traces
    col = babeltrace.TraceCollection()

    # Add the trace provided by the user (LTTng traces always have
    # the 'ctf' format)
    if col.add_trace(trace_file, 'ctf') is None:
        raise RuntimeError('Cannot add trace')

    names = set()
    commands = set()
    event_count = 0
    for event in col.events:
        event_count += 1
        names.add(event.name)
        if event.name == 'sched_switch':
            commands.add(event['prev_comm'])

    i_to_name = dict(enumerate(names))
    name_to_i = {v: k for k, v in i_to_name.items()}

    i_to_comm = dict(enumerate(commands))
    comm_to_i = {v: k for k, v in i_to_comm.items()}

    pickle.dump(i_to_name, open(save_dir+'i2n.p', 'wb'))
    pickle.dump(i_to_comm, open(save_dir+'i2c.p', 'wb'))


    cur_comms = {}

    window_count = np.floor(event_count / window_size).astype(int)
    windows = np.zeros([window_count, len(commands), len(names)], int)
    cur_window = 0
    window_event_count = 0
    # Iterate on events
    for event in col.events:
        window_event_count += 1
        if window_event_count > window_size:
            window_event_count = 1
            cur_window += 1
            if cur_window >= window_count:
                break

        cur_cpu = event['cpu_id']
        # if switch, update current task on cpu
        if event.name == 'sched_switch':
            cur_comms[cur_cpu] = event['next_comm']
            continue

        comm = cur_comms[cur_cpu]
        # if we don't know the current command, continue
        if comm is None:
            continue

        windows[cur_window][comm_to_i.get(comm)][name_to_i.get(event.name)] += 1

    windows = windows.reshape([window_count, len(commands) * len(names)])

    pickle.dump(windows, open(save_dir+'windows.p', 'wb'))
    return windows

if __name__ == '__main__':
    sys.exit(0 if parse(trace_file=sys.argv[1]).any() else 1)
