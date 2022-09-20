class Singleton(type):
    
    __instances = {}
    
    def __new__(self, name, bases, attrs):
        return super().__new__(self, name, bases, attrs)
    
    def __init__(cls, *args, **kwargs):
        super(Singleton, cls).__init__(*args, **kwargs)
        
    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            return cls.__instances[cls]
        else:
            return cls.__instances[cls]