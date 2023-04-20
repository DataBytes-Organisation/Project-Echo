import datetime
import time

class Clock:
    
    # track the singleton instance of the clock
    _instance = None

    def __new__(cls, step_interval = 0.2, start_time=datetime.datetime.now()):
        if cls._instance is None:
            print('Creating the Clock Once')
            cls._instance = super(Clock, cls).__new__(cls)
            cls.current_time = start_time
            cls.step_interval = step_interval
            cls.last_sync_time = datetime.datetime.now()
        return cls._instance
        
    def get_time(self):
        return self.current_time
    
    def advance_time(self, delta, unit='seconds'):
        if unit == 'days':
            self.current_time += datetime.timedelta(days=delta)
        elif unit == 'microseconds':
            self.current_time += datetime.timedelta(microseconds=delta)
        else:  # default to seconds
            self.current_time += datetime.timedelta(seconds=delta)
        
    # step the clock, intended to be called in the simulation loop  
    def update(self):
        self.advance_time(self.step_interval*1000000.0, 'microseconds')
        print(f'\n\nsimulated time is now: ', self.current_time)
            
    # supporting the simulation loop, intended to be called at the end of the loop
    def wait_real_time_sync(self):
        
        # calculate how much wall clock time elapsed since last call
        timediff = (datetime.datetime.now() - self.last_sync_time).total_seconds()

        # calculate how much wall clock time we should wait so sim runs real time
        waittime = self.step_interval - timediff
    
        # print(f'waittime: {waittime}')
        
        if waittime > 0:
            time.sleep(waittime)
            
        # record the last time this sync function was called
        self.last_sync_time = datetime.datetime.now()    
    
    def test(self):
        
        # create a clock with wall time now
        clock = Clock(0.2)
        
        print(f'sim time is: {clock.get_time()}')
        
        clock.advance_time(200000, 'microseconds')
        
        print(f'sim time is: {clock.get_time()}')

