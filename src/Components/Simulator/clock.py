import datetime

class Clock:
    def __init__(self, step_interval = 200, start_time=datetime.datetime.now()):
        self.current_time = start_time
        self.step_interval = step_interval
    
    def get_time(self):
        return self.current_time
    
    def advance_time(self, delta, unit='seconds'):
        if unit == 'days':
            self.current_time += datetime.timedelta(days=delta)
        elif unit == 'microseconds':
            self.current_time += datetime.timedelta(microseconds=delta)
        else:  # default to seconds
            self.current_time += datetime.timedelta(seconds=delta)
            
    def wait_real_time_sync(self):
        # TODO - implementation not complete
        self.last_sync_time = datetime.datetime.now()
        pass
    
    def test(self):
        
        # create a clock with wall time now
        clock = Clock()
        
        print(f'sim time is: {clock.get_time()}')
        
        clock.advance_time(200000, 'microseconds')
        
        print(f'sim time is: {clock.get_time()}')

