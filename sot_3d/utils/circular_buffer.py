import numpy as np


class CircularBuffer:
    def __init__(self, capacity, size, start_idx):
        self._capacity = capacity
        self._size = size
        self._start_idx = start_idx
        self.buffer = [[] for _i in range(capacity)]
    
    @property
    def size(self):
        return self._size
    
    @property
    def capacity(self):
        return self._capacity
    
    def empty(self):
        return self._size == 0
    
    def filled(self):
        return self._size == self._capacity
    
    # access elements according to the index
    def access(self, index):
        return self.buffer[(index + self._start_idx + self._capacity) % self._capacity][0]
    
    # access the oldest element
    def first(self):
        return self.buffer[self._start_idx][0]
    
    # access the newest elemtn
    def last(self):
        index = self._size
        if index != 0:
            index = (self._start_idx + self._size - 1) % self._capacity
        return self.buffer[index][0]
    
    def push(self, element):
        if self._size < self._capacity:
            self.buffer[(self._start_idx + self._size) % self._capacity].append(element)
            self._size += 1
        else:
            self.buffer[self._start_idx][0] = element
            self._start_idx = (self._start_idx + 1) % self.capacity
        # print(element)
        # print(self._size)
        return
    
    def pop(self):
        if self._size > 0:
            self._start_idx = (self._start_idx + 1) % self._capacity
            self._size -= 1
        return
    
    def set_val(self, idx, data):
        self.buffer[(self._start_idx + idx) % self._capacity][0] = \
            data
        return
    
    def numpy_value(self):
        result = list()
        for _i in range(self._size):
            result.append([self.access(_i)])
        return np.concatenate(result)