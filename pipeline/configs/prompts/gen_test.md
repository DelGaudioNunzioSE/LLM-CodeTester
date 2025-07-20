I need a coding assistant specialized in generating standalone test functions with `assert` statements (no classes)
You do not know the solution function implementation.
Please write exactly 3 standalone test functions.


### Example:
#### Problem:
Return the sum of two integers.

#### Solution Function Signature Only:
```python
def add(a, b):
```

#### output test:
```python
from solution import add

def test_add_basic():
    assert add(1, 2) == 3

def test_add_zero():
    assert add(2, 0) == 2

def test_add_negative():
    assert add(-1, 2) == 1
```


### Input:
#### Problem Description:
{description}

#### Solution Function Signature Only:
```python
{code}
```

