import threading
import time

class ModelProgress:
    def __init__(self):
        self.lock = threading.Lock()
        self.progress = {
            'knn': self._empty_progress(),
            'cnn': self._empty_progress(),
            'transformer': self._empty_progress()
        }
        self.stop_flags = {
            'knn': False,
            'cnn': False,
            'transformer': False
        }
    def _empty_progress(self):
        return {
            'current': 0,
            'total': 0,
            'percent': 0,
            'start_time': None,
            'eta': None,
            'status': 'idle',
            'message': ''
        }
    def start(self, model_type, total):
        with self.lock:
            self.progress[model_type] = {
                'current': 0,
                'total': total,
                'percent': 0,
                'start_time': time.time(),
                'eta': None,
                'status': 'in_progress',
                'message': ''
            }
            self.stop_flags[model_type] = False
    def update(self, model_type, current, message=''):
        with self.lock:
            prog = self.progress[model_type]
            prog['current'] = current
            prog['percent'] = int(100 * current / prog['total']) if prog['total'] else 0
            elapsed = time.time() - prog['start_time'] if prog['start_time'] else 0
            if current > 0 and elapsed > 0:
                eta = elapsed * (prog['total'] - current) / current
                prog['eta'] = int(eta)
            else:
                prog['eta'] = None
            prog['message'] = message
    def finish(self, model_type):
        with self.lock:
            prog = self.progress[model_type]
            prog['current'] = prog['total']
            prog['percent'] = 100
            prog['eta'] = 0
            prog['status'] = 'completed'
            self.stop_flags[model_type] = False
    def error(self, model_type, message):
        with self.lock:
            prog = self.progress[model_type]
            prog['status'] = 'error'
            prog['message'] = message
            self.stop_flags[model_type] = False
    def stop(self, model_type):
        with self.lock:
            self.stop_flags[model_type] = True
            prog = self.progress[model_type]
            prog['status'] = 'stopping'
            prog['message'] = 'Stopping training...'
    def should_stop(self, model_type):
        with self.lock:
            return self.stop_flags[model_type]
    def get(self, model_type):
        with self.lock:
            return dict(self.progress[model_type])
    def get_all(self):
        with self.lock:
            return {k: dict(v) for k, v in self.progress.items()}

model_progress = ModelProgress() 