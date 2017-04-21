import os.path

import babeltrace
import numpy as np

import pickle


class trace_parser:
    i_to_syscall_filename = 'i2s.p'
    i_to_process_filename = 'i2p.p'
    windows_filename = 'win.p'
    windows = []
    i_to_syscall = {}
    syscall_to_i = {}
    i_to_process = {}
    process_to_i = {}
    syscalls_start_idx = 0

    def __init__(self, trace_file, trace_name, window_size=250, simple=True, save_dir=None):
        self.trace_file = trace_file
        self.trace_name = trace_name
        self.window_size = window_size
        self.simple = simple
        if save_dir is None:
            simple_text = 'full'
            if simple:
                simple_text = 'simple'

            self.save_dir = os.path.join('pickle/', trace_name, simple_text, str(window_size))
        else:
            self.save_dir = save_dir
        self.parse()


    def parse(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        windows_path = os.path.join(os.path.join(self.save_dir, trace_parser.windows_filename))
        i_to_syscall_path = os.path.join(self.save_dir, trace_parser.i_to_syscall_filename)
        i_to_process_path = os.path.join(self.save_dir, trace_parser.i_to_process_filename)
        if os.path.isfile(os.path.join(windows_path)):
            self.windows = pickle.load(open(windows_path, 'rb'))
            self.i_to_syscall = pickle.load(open(i_to_syscall_path, 'rb'))
            self.i_to_process = pickle.load(open(i_to_process_path, 'rb'))
            self.syscall_to_i = {v: k for k, v in self.i_to_syscall.items()}
            self.process_to_i = {v: k for k, v in self.i_to_process.items()}
            self.syscalls_start_idx = len(self.i_to_process)
            return self.windows

        # A trace collection contains one or more traces
        col = babeltrace.TraceCollection()

        # Add the trace provided by the user (LTTng traces always have
        # the 'ctf' format)
        if col.add_trace(self.trace_file, 'ctf') is None:
            raise RuntimeError('Cannot add trace')

        syscalls = set()
        processes = set()
        event_count = 0
        for event in col.events:
            event_count += 1
            syscalls.add(event.name)
            if event.name == 'sched_switch':
                processes.add(event['prev_comm'])

        self.syscalls_start_idx = len(processes)

        self.i_to_process = dict(enumerate(processes))
        self.process_to_i = {v: k for k, v in self.i_to_process.items()}

        if self.simple:
            self.i_to_syscall = {idx + self.syscalls_start_idx: syscall for idx, syscall in enumerate(syscalls)}
        else:
            self.i_to_syscall =  dict(enumerate(syscalls))

        self.syscall_to_i = {v: k for k, v in self.i_to_syscall.items()}


        pickle.dump(self.i_to_syscall, open(i_to_syscall_path, 'wb'))
        pickle.dump(self.i_to_process, open(i_to_process_path, 'wb'))

        current_processes = {}

        window_count = np.floor(event_count / self.window_size).astype(int)

        if self.simple:
            windows = np.zeros([window_count, len(processes) + len(syscalls)], int)
        else:
            windows = np.zeros([window_count, len(processes), len(syscalls)], int)

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

            if self.simple:
                windows[cur_window][self.syscall_to_i.get(event.name)] += 1

            cur_cpu = event['cpu_id']
            # if switch, update current task on cpu
            if event.name == 'sched_switch':
                current_processes[cur_cpu] = event['next_comm']
                continue

            process = current_processes[cur_cpu]
            if process is None:
                continue

            if self.simple:
                windows[cur_window][self.process_to_i.get(process)] += 1
            else:
                windows[cur_window][self.process_to_i.get(process)][self.syscall_to_i.get(event.name)] += 1

        # Normalize
        self.windows = np.divide(windows, self.window_size)
        if not self.simple:
            self.windows = self.windows.reshape([window_count, len(processes) * len(syscalls)])

        pickle.dump(self.windows, open(windows_path, 'wb'))
        return self.windows

    def window_to_string(self, window, separator='\n'):
        if self.simple:
            return self.__window_to_string_simple__(window, separator)
        return self.__window_to_string_full__(window, separator)

    def __window_to_string_full__(self, window, separator='\n'):
        description = ''
        window = np.multiply(window, self.window_size).astype(int)
        window_shaped = window.reshape([len(self.process_to_i), len(self.syscall_to_i)])
        for i,names in enumerate(window_shaped):
            used_names = names > 0
            if any(used_names):
                description += str(self.i_to_process[i]) + separator
                used_names_idx = np.where(used_names)
                for name_idx  in used_names_idx[0]:
                    description += str(self.i_to_syscall[name_idx]) + ": " + str(names[name_idx]) + '; '
                description += separator
        return description

    def __window_to_string_simple__(self, window, separator):
        description = ''
        window = np.rint(np.multiply(window, self.window_size)).astype(int)
        for idx,count in enumerate(window):
            if count == 0:
                continue
            if idx >= self.syscalls_start_idx:
                description += self.i_to_syscall[idx] + ": " + str(count) + separator
            else:
                description += self.i_to_process[idx] + ": " + str(count) + separator
        return description


if __name__ == '__main__':
    trace_p = trace_parser('/home/kfedorov/lttng-traces/firefox/kernel/', 'firefox', simple=False)
    print(trace_p.window_to_string(trace_p.windows[200]))