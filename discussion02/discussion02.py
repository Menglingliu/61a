# result = (lambda x: 2 * (lambda x: 3)(4) * x)(5)
# print(result)

# Q2: make keeper

def make_keeper(n):
    """Returns a function that takes one parameter cond and prints
    out all integers 1..i..n where calling cond(i) returns True.

    >>> def is_even(x): # Even numbers have remainder 0 when divided by 2.
    ...     return x % 2 == 0
    >>> make_keeper(5)(is_even)
    2
    4
    >>> make_keeper(5)(lambda x: True)
    1
    2
    3
    4
    5
    >>> make_keeper(5)(lambda x: False)  # Nothing is printed
    """
    "*** YOUR CODE HERE ***"

    def func(cond):
        i = 1
        while i<=n:
            if cond(i):
                print(i)
            i += 1
    return func


def is_even(x): # Even numbers have remainder 0 when divided by 2.
    return x % 2 == 0

# make_keeper(5)
# make_keeper(5)(is_even)
# make_keeper(5)(lambda x: True)
# make_keeper(5)(lambda x: False)  # Nothing is printed


# Q3: Digit Finder

def find_digit(k):
    """Returns a function that returns the kth digit of x.

    >>> find_digit(2)(3456)
    5
    >>> find_digit(2)(5678)
    7
    >>> find_digit(1)(10)
    0
    >>> find_digit(4)(789)
    0
    """
    assert k > 0
    "*** YOUR CODE HERE ***"
    
    def func(n):
        return n % pow(10, k) // pow(10, k-1)
    return func

#print(find_digit(2)(3456))

print(find_digit(2)(3456))
print(find_digit(2)(5678))
print(find_digit(1)(10))
print(find_digit(4)(789))


def find_digit_1(k):
    """Returns a function that returns the kth digit of x.

    >>> find_digit(2)(3456)
    5
    >>> find_digit(2)(5678)
    7
    >>> find_digit(1)(10)
    0
    >>> find_digit(4)(789)
    0
    """
    assert k > 0
    "*** YOUR CODE HERE ***"
    
    return lambda n: n % pow(10, k) // pow(10, k-1)

print(find_digit_1(2)(3456))
print(find_digit_1(2)(5678))
print(find_digit_1(1)(10))
print(find_digit_1(4)(789))


# Q4 Match maker

def match_k(k):
    """Returns a function that checks if digits k apart match.

    >>> match_k(2)(1010)
    True
    >>> match_k(2)(2010)
    False
    >>> match_k(1)(1010)
    False
    >>> match_k(1)(1)
    True
    >>> match_k(1)(2111111111111111)
    False
    >>> match_k(3)(123123)
    True
    >>> match_k(2)(123123)
    False
    """
    def check(x):
        while x // (10 ** k) > 0:
            if (x % 10) != (x // (10 ** k)) % 10:
                return False
            x //= 10
        return True
    return check


print(match_k(2)(1010))
print(match_k(2)(2010))
print(match_k(1)(1010))
print(match_k(1)(1))
print(match_k(1)(2111111111111111))
print(match_k(3)(123123))
print(match_k(2)(123123))

