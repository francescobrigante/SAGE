from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple
import torch.nn as nn

class AbstractEncoder(nn.Module, ABC):
    def __init__(self, *, input_size: int, is_complex: bool):
        super().__init__()
        self.input_size = input_size
        self.is_complex = is_complex
        
class AbastractDecoder(nn.Module, ABC):
    def __init__(self, *, channels: int, is_complex: bool):
        super().__init__()
        self.channels = channels
        self.is_complex = is_complex
        
    
        

    