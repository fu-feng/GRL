import multiprocessing as mp

class Global_rewards:
    def __init__(self, info_manager):
        self.info_lock = info_manager.Lock()
        self.info_dict = info_manager.dict()
    
    def set_value(self, name, value):
        self.info_lock.acquire()
        self.info_dict[name] = value
        self.info_lock.release()
    
    def get_dict(self):
        # global info_lock, info_dict
        return self.info_dict
