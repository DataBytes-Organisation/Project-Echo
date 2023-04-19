import datetime

class Clock:
    def __init__(self, start_time=datetime.datetime.now()):
        self.current_time = start_time
    
    def get_time(self):
        return self.current_time
    
    def advance_time(self, delta, unit='seconds'):
        if unit == 'days':
            self.current_time += datetime.timedelta(days=delta)
        elif unit == 'microseconds':
            self.current_time += datetime.timedelta(microseconds=delta)
        else:  # default to seconds
            self.current_time += datetime.timedelta(seconds=delta)

