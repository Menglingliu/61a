# Q1 Big fib

def gen_fib():
    n, add = 0, 1
    while True:
        yield n
        n, add = n + add, n

# (lambda x: x*x) (4)

# def f(x):
#     return x * x

# def f(fun):
#     return fun(x)

(lambda t: [next(t) for _ in range(10)])(gen_fib())

# it evaluates to the smallest Fibonacci number that is larger than 2024
next(filter(lambda n: n > 2024, gen_fib()))


# Q2 Something different

def differences(t):
    """Yield the differences between adjacent values from iterator t.

    >>> list(differences(iter([5, 2, -100, 103])))
    [-3, -102, 203]
    >>> next(differences(iter([39, 100])))
    61
    """
    "*** YOUR CODE HERE ***"
    previous_x = 0
    i = 0
    for x in t:
        if i != 0:
            yield x - previous_x
        previous_x = x
        i += 1

def differences_1(t):
    previous_x = next(t)
    for x in t:
        yield x - previous_x
        previous_x = x



# Q3 Partitions

def partition_gen(n, m):
    """Yield the partitions of n using parts up to size m.

    >>> for partition in sorted(partition_gen(6, 4)):
    ...     print(partition)
    1 + 1 + 1 + 1 + 1 + 1
    1 + 1 + 1 + 1 + 2
    1 + 1 + 1 + 3
    1 + 1 + 2 + 2
    1 + 1 + 4
    1 + 2 + 3
    2 + 2 + 2
    2 + 4
    3 + 3
    """
    assert n > 0 and m > 0
    if n == m:
        yield str(m)
    if n - m > 0:
        for p in partition_gen(n-m, m):
            yield p + ' + ' + str(m)
    if m > 1:
        yield from partition_gen(n, m-1)
        











