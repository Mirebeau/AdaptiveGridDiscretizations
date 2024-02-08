# Code automatically exported from notebook Time1D_NonDiv.ipynb in directory Notebooks_NonDiv
# Do not modify
#from itertools import accumulate # Accumulate with initial value only exists in Python >= 3.8
def accumulate(iterable, func, initial):
    yield initial
    for element in iterable:
        initial = func(initial, element)
        yield initial

