class NotAnArray(object):
    " An object that is convertable to an array"

    def __init__(self, data):
        self.data = data

    def __array__(self, dtype=None):
        return self.data

