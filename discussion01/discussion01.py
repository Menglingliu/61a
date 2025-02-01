
def fizzbuzz(n):
    """
    >>> result = fizzbuzz(16)
    1
    2
    fizz
    4
    buzz
    fizz
    7
    8
    fizz
    buzz
    11
    fizz
    13
    14
    fizzbuzz
    16
    >>> print(result)
    None
    """
    "*** Your code here ***"

    i = 0
    while i <= n:
        if i % 3 == 0 & i % 5 == 0:
            print('fizzbuzz')
        elif i % 3 == 0:
            print('fizz')
        elif i % 5 == 0:
            print('buzz')
        else: 
            print(i)
        i += 1

# result1 = fizzbuzz(16)
# print(result1)


def is_prime(n):
    """
    >>> is_prime(10)
    False
    >>> is_prime(7)
    True
    >>> is_prime(1) # one is not a prime number!!
    False
    """

    "*** Your code here ***"
    
    # *** for loop ***
    if n == 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
    
    # *** while loop ***
    # if n == 1:
    #     return False
    # k = 2
    # while k < n:
    #     if n % k == 0:
    #         return False
    #     k += 1
    # return True

# result2 = is_prime(2)
# print(result2)


def has_digit(n, k):
    """Returns whether k is a digit in n.

    >>> has_digit(10, 1)
    True
    >>> has_digit(12, 7)
    False
    """
    assert k >= 0 and k < 10
    "*** YOUR CODE HERE ***"

    while n > 0:
        last = n % 10
        n = n // 10
        if last == k:
            return True
    return False


result = has_digit(12,2)
print(result)

def unique_digits(n):
    """Return the number of unique digits in positive integer n.

    >>> unique_digits(8675309) # All are unique
    7
    >>> unique_digits(13173131) # 1, 3, and 7
    3
    >>> unique_digits(101) # 0 and 1
    2
    """
    "*** YOUR CODE HERE ***"
    
    unique = 0
    while n > 0:
        last = n % 10
        n = n // 10
        if not has_digit(n, last):
            unique += 1
    return unique

result = unique_digits(10333394555658)
print(result)