from collections import OrderedDict

class ATimeCache(object):
    """Cache class (dictionary) with a limited size, where only the
    'max_entries' most recently added or accessed entries are stored."""

    def __init__(self, max_entries):
        self._cache = OrderedDict()
        self._max_entries = max_entries

    def _shrink(self):
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        
    def get_max_entries(self):
        return self._max_entries

    def set_max_entries(self, value):
        self._max_entries = value
        self._shrink()
        
    max_entries = property(
        get_max_entries, set_max_entries, None, "Set or get the cache size")

    def has_key(self, key):
        return self._cache.has_key(key)
    
    def __eq__(self, other):
        try:
            return self._cache.__eq__(other._cache)
        except:
            return False
    
    def __len__(self):
        return self._cache.__len__()

    def __getitem__(self, key):
        value = self._cache.pop(key)
        self._cache[key] = value
        return value
        
    def __setitem__(self, key, value):
        if self._cache.has_key(key):
            self._cache.pop(key)
        self._cache.__setitem__(key, value)
        self._shrink()

    def __contains__(self, key):
        return self.has_key(key)

    def __str__(self):
        return self.cache.__str__()

if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
