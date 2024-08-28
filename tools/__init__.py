import sys
from os.path import dirname

ROOT = dirname(dirname(sys.modules[__name__].__file__))
sys.path.append(ROOT)