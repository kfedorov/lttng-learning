from collections import Counter
import babeltrace
import sys
import numpy as np
import pickle
import os.path


class traceParser:
    windows = []
    i_to_name = {}
    name_to_i = {}
    i_to_comm = {}
    comm_to_i = {}
    names_start_idx = 0

    def __init__(self, trace_file, trace_name, window_size=250):
        self.trace_file = trace_file
        self.trace_name = trace_name
        self.window_size = window_size
        self.save_dir = 'pickle/simple/' + trace_name + '/'
        self.parse()


    def parse(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if os.path.isfile(self.save_dir+'windows.p'):
            self.windows = pickle.load(open(self.save_dir+'windows.p', 'rb'))
            self.i_to_name = pickle.load(open(self.save_dir+'i2n.p', 'rb'))
            self.i_to_comm = pickle.load(open(self.save_dir+'i2c.p', 'rb'))
            self.name_to_i = {v: k for k, v in self.i_to_name.items()}
            self.comm_to_i = {v: k for k, v in self.i_to_comm.items()}
            self.names_start_idx = len(self.i_to_comm)
            return self.windows

        # A trace collection contains one or more traces
        col = babeltrace.TraceCollection()

        # Add the trace provided by the user (LTTng traces always have
        # the 'ctf' format)
        if col.add_trace(self.trace_file, 'ctf') is None:
            raise RuntimeError('Cannot add trace')

        names = set()
        commands = set()
        event_count = 0
        for event in col.events:
            event_count += 1
            names.add(event.name)
            if event.name == 'sched_switch':
                commands.add(event['prev_comm'])

        self.names_start_idx = len(commands)

        self.i_to_comm = dict(enumerate(commands))
        self.comm_to_i = {v: k for k, v in self.i_to_comm.items()}

        self.i_to_name = {idx + self.names_start_idx: name for idx, name in enumerate(names)}
        self.name_to_i = {v: k for k, v in self.i_to_name.items()}

        pickle.dump(self.i_to_name, open(self.save_dir+'i2n.p', 'wb'))
        pickle.dump(self.i_to_comm, open(self.save_dir+'i2c.p', 'wb'))

        cur_comms = {}

        window_count = np.floor(event_count / self.window_size).astype(int)
        windows = np.zeros([window_count, len(commands) + len(names)], int)
        cur_window = 0
        window_event_count = 0
        # Iterate on events
        for event in col.events:
            window_event_count += 1
            if window_event_count > self.window_size:
                window_event_count = 1
                cur_window += 1
                if cur_window >= window_count:
                    break

            windows[cur_window][self.name_to_i.get(event.name)] += 1

            cur_cpu = event['cpu_id']
            # if switch, update current task on cpu
            if event.name == 'sched_switch':
                cur_comms[cur_cpu] = event['next_comm']
                continue

            comm = cur_comms[cur_cpu]
            # if we don't know the current command, continue
            if comm is None:
                continue

            windows[cur_window][self.comm_to_i.get(comm)] += 1

        # Normalize
        self.windows = np.divide(windows, self.window_size)

        pickle.dump(self.windows, open(self.save_dir+'windows.p', 'wb'))
        return self.windows

    def windowToString(self, window, separator='\n'):
        description = ''
        window = np.multiply(window, self.window_size).astype(int)
        for idx,count in enumerate(window):
            if count == 0:
                continue
            if idx >= self.names_start_idx:
                description += self.i_to_name[idx] + ": " + str(count) + separator
            else:
                description += self.i_to_comm[idx] + ": " + str(count) + separator

        return description


if __name__ == '__main__':
    traceP = traceParser('/home/kfedorov/lttng-traces/firefox/kernel/', 'firefox')
    print(traceP.windowToString(traceP.windows[254]))