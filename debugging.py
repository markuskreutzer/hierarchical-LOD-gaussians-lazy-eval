import time
from collections import deque

class timestamp_wrapper:
    def __init__(self, ringbuffer_size=10):
        self.filter_phase_begin = None
        self.load_phase_begin = None
        self.render_phase_begin = None
        self.pass_end = None
        
        self.filter_times = deque(maxlen=ringbuffer_size)
        self.load_times = deque(maxlen=ringbuffer_size)
        self.render_times = deque(maxlen=ringbuffer_size)
        self.total_times = deque(maxlen=ringbuffer_size)
        
        pass

    def stamp(self, mode):
        if mode == 'filter':
            self.filter_phase_begin = time.time_ns()
        elif mode == 'load':
            self.load_phase_begin = time.time_ns()
        elif mode == 'render':
            self.render_phase_begin = time.time_ns()
        else:
            self.pass_end = time.time_ns()

    def get_timings(self):
        if self.filter_phase_begin == None or self.load_phase_begin == None or self.render_phase_begin == None or self.pass_end == None:
            return "NOT ALL TIMESTAMPS SET!"

        filter_ms = (self.load_phase_begin - self.filter_phase_begin)/1000000
        load_ms = (self.render_phase_begin - self.load_phase_begin)/1000000
        render_ms = (self.pass_end - self.render_phase_begin)/1000000
        total_ms = (self.pass_end - self.filter_phase_begin)/1000000

        self.filter_times.append(filter_ms)
        self.load_times.append(load_ms)
        self.render_times.append(render_ms)
        self.total_times.append(total_ms)

        return (
            f'FILTER={filter_ms:.3f}ms ({filter_ms / total_ms:.1%}) | '
            f'LOAD={load_ms:.3f}ms ({load_ms / total_ms:.1%}) | '
            f'RENDER={render_ms:.3f}ms ({render_ms / total_ms:.1%}) | '
            f'TOTAL={total_ms:.3f}ms'
        )
    
    def get_means(self):
        if not self.total_times:
            return "NO TIMINGS RECORDED YET!"

        mean_filter = sum(self.filter_times) / len(self.filter_times)
        mean_load = sum(self.load_times) / len(self.load_times)
        mean_render = sum(self.render_times) / len(self.render_times)
        mean_total = sum(self.total_times) / len(self.total_times)

        return (
            f'FILTER={mean_filter:.3f}ms ({mean_filter / mean_total:.1%}) | '
            f'LOAD={mean_load:.3f}ms ({mean_load / mean_total:.1%}) | '
            f'RENDER={mean_render:.3f}ms ({mean_render / mean_total:.1%}) | '
            f'TOTAL={mean_total:.3f}ms [avg over {len(self.total_times)} iterations]'
        )