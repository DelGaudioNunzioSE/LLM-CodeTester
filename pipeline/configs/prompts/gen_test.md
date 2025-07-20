## System
Please be an assistant that writes Pytest-style unit tests for Python functions, based on problem description and the . The tests should be hard to pass, meaning they only succeed if the implementation is robust and free of subtle bugs.

## User
Write Pytest-style unit tests for the following Python function.

Use only standalone test functions with `assert` statements (no classes). Base your tests on the expected behavior described in the problem.


### Example:
#### Problem:
Return the sum of two integers.

#### Function:
```python
def add(a, b):
    sum = a + b
    return sum
```

#### Test:
```python
from solution import add

def test_add_basic():
    assert add(0, 0) == 0

def test_add_zero():
    assert add(-5, 6) == 1
```


### Input:
### Problem Description:
{description}

### Function:
```python
{code}
```

### Output: