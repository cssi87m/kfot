import sys
import torchvision.transforms.functional as F
import time

def a():
    return None

s = time.time()
print(getattr(
    None, "a"
))
print(time.time() - s)