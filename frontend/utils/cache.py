import hashlib
import json
from typing import Any, Dict, Optional
from functools import wraps

def hash_input_data(data: Dict[str, Any]) -> str:
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

class SimpleCache:
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        self._cache[key] = value
    
    def get_model(self, key: str) -> Optional[Any]:
        return self.get(key)
    
    def put_model(self, key: str, value: Any) -> None:
        self.put(key, value)
    
    def clear(self) -> None:
        self._cache.clear()

_model_cache = SimpleCache()

def get_model_cache() -> SimpleCache:
    return _model_cache

def cache_prediction():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
