"""Topic modelling and entity analysis for NewsPrism."""
from .pipeline import Result, extract_entities, run, write

__all__ = ['Result', 'extract_entities', 'run', 'write']
__version__ = '1.0.0'
