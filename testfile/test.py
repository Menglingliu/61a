def apply_twice(f,x):
    return f(f(x))

def square(x):
    return x * x

square = lambda x: x * x
square(4)

#(lambda x:x*x)(3)

def sum_naturals(n):
    total = 0
    k = 1
    while k <= n:
        total += k
        k += 1
    return total

#print(sum_naturals(100))


def curried_pow(x):
    def h(y):
        return pow(x,y)
    return h


def map_to_tange(start, end, f):
    while start < end:
        print(f(start))
        start = start + 1


# def trace(fn):
#     def wrapped(x):
#         print('-> ', fn, '(', x, ')')
#         return fn(x)
#     return wrapped

# def triple(x):
#     return 3 * x

# # result = trace(triple)(10)
# # print(result)

# @trace
#     def triple(x):
#         return 3 * x


def end(n,d):
    while n > 0:
        last = n % 10
        n = n // 10
        print(last)
        if d == last:
            break

def end(n,d):
    last = None
    while n > 0 and d != last:
        last = n % 10
        n = n // 10
        print(last)
        # if d == last:
        #     break

def end(n,d):
    while True:
        last = n % 10
        n = n // 10
        print(last)
        if n <= 0 or d == last:
            break

def end(n,d):
    while n > 0:
        last = n % 10
        n = n // 10
        print(last)
        if d == last:
            return None


# print(end(354354,5))

# Returns:
# a higher order function f which takes an integer x starting at 0 and working its way up until 
# it founds one where f(x) is a true value. Keep trying x by construct an infinite loops.
# This infinite loop will run forever unless a return statement within it tells us we are done.
def search(f):
    x = 0
    while True: # construct an infinite loop
        if f(x):
            return x
        x += 1

# equivalent version:
def search(f):
    x = 0
    while not f(x):
        x += 1
    return x

def positive(x):
    return max(0, square(x) - 100)

print(search(positive))

def inverse(f):
    return lambda y: search(lambda x: f(x) == y)

sqrt = inverse(square)
print(square(30))
print(sqrt(900))

def scale(f, x, k):
    return k * f(x)

print(scale(square, 3, 2))

print(scale(lambda x:x+10, 5, 2))

def multiply_by(m):
    def multiply(n):
        return n * m
    return multiply

times_three = multiply_by(3)
print(times_three(5))
print(multiply_by(3)(5))

x = 10
y = x
x = 20
x, y = y + 1, x -1
#y = x - 1
print(x)
print(y)


a = lambda x: x
print(a(5))


b = lambda x, y: lambda: x + y  # Lambdas can return other lambdas!
c = b(8, 4)
print(c)
print(c())

print((lambda: 3)())


d = lambda f: f(4)
def square(x):
    return x * x

print(d(square))

call_thrice = lambda f: lambda x: f(f(f(x)))
result = call_thrice(lambda y: y+1)(0)
print(result)

print_lambda = lambda z:print(z)
print_lambda

one_throusand = print_lambda(1000)
one_throusand


def trace1(fn):
    def traced(x):
        print('Calling', fn, 'on argument', x)
        return fn(x)
    return traced

@trace1
def square(x):
    return x*x

@trace1
def sum_squares_up_to(n):
    k = 1
    total = 0
    while k<=n:
        total = total + square(k)
        k = k + 1
    return total

trace1(square)(12)


def split(n):
    return n // 10, n % 10

def sum_digits(n):
    if n < 10:
        return n
    else:
        all_but_last, last = split(n)
    return sum_digits(all_but_last) + last

print(sum_digits(1234))
print(sum_digits(123433534))


# mutual recursion
def split(n):
    return n//10, n%10

def sum_digits(n):
    if n < 10:
        return n
    else: 
        all_but_last, last = split(n)
        return sum_digits(all_but_last) + last

def luhn_sum(n):
    if n < 10:
        return n
    else:
        all_but_last, last = split(n)
        return luhn_sum_double(all_but_last) + last

def luhn_sum_double(n):
    if n < 10:
        return sum_digits(2*n)
    else: 
        all_but_last, last = split(n)
        return luhn_sum(all_but_last) + sum_digits(2*last)

print(luhn_sum(5105105105105100))
print(luhn_sum(32))


def is_even(n):
    if n == 0:
        return True
    else:
        return is_odd(n-1)

def is_odd(n):
    if n == 0:
        return False
    else:
        return is_even(n-1)

result = is_even(4)
print(result)

    
def is_even(n):
    if n == 0:
        return True
    else:
        if (n-1) == 0:
            return False
        else:
            return is_even((n-1)-1)

def cascade(n):
    if n < 10:
        print(n) # base case
    else:
        print(n)
        cascade(n//10)
        print(n)


def cascade(n):
    print(n)
    if n >= 10:
        cascade(n // 10)
        print(n)


def inverse_cascade(n):
    grow(n)
    print(n)
    shrink(n)

def f_then_g(f, g, n):
    if n:
        f(n)
        g(n)

grow = lambda n: f_then_g(grow, print, n//10)
shrink = lambda n: f_then_g(print, shrink, n//10)


def grow_0(n):
    if n//10:
        grow_0(n//10)
        print(n//10)

def shrink_0(n):
    if n//10:
        print(n//10)
        shrink_0(n//10)

def grow_1(n):
    if n < 10:
        print(n)
    else:
        grow_1(n//10)
        print(n)

def shrink_1(n):
    if n < 10:
        print(n)
    else:        
        print(n)
        shrink_1(n//10)

# fibonaci sequence
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-2) + fib(n-1)


def count_partitions(n, m):
    if n == 0:
        return 1
    elif n < 0:
        return 0
    elif m == 0:
        return 0
    else:
        with_m = count_partitions(n-m, m)
        without_m = count_partitions(n, m-1)
        return with_m + without_m


# Baobao's method
def count_partitions(n, m):
    if m == 1 or n == 1:
        return 1
    base = 0
    cnt = n // m
    for i in range(cnt + 1):
        base += count_partitions(n - i * m, m - 1)
    return base


def count(s, value):
    total, index = 0,0
    while index < len(s):
        element = s[index]
        if element == value:
            total += 1
        index += 1
    return total

def count(s, value):

    total = 0
    for element in s:
        if element == value:
            total += 1
    return total 

def sum_below(n):
    total = 0
    for i in range(n):
        total += i
    return total

def cheer():
    for _ in range(3):
        print('Go Bears!')


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'm', 'n', 'o', 'p']
[letters[i] for i in [3,4,6,8]] # return a list that contains ith elemtn in this above list

odds = [1,3,5,7,9]
[x+1 for x in odds]

[x for x in odds if 25 % x == 0]

def divisors(n):
    return [1] + [x for x in range(2, n) if n % x == 0]

from operator import sub, mul

fact = lambda n: 1 if n == 1 else mul(n, fact(n-1))
fact(5)


def count(s, value):
    total = 0
    for elem in s:
        if elem == value:
            total = total +1
    return total 

numerals = {'I': 1, 'V': 5, 'X': 10}

def index(keys, values, match):
    """
    >>> index([7,9,11], range(30,50), lambda k,v: v % k == 0)
    """
    return {k:[v for v in values if match(k,v)] for k in keys}


def rational(n,d):
    def select(name):
        if name == 'n':
            return n 
        elif name == 'd':
            return d
    return select

x = rational(3,4)

def numer(x):
    return x('n')

def denom(x):
    return x('d')


def pair(x, y):
    """Return a function that represents a pair."""
    def get(index):
        if index == 0:
            return x
        elif index == 1:
            return y
    return get

def select(p, i):
    """Return the element at index i of pair p."""
    return p(i)      


# ---------- TREE ---------- #

# Define a tree

# constructor
def tree(label, branches = []):
    for branch in branches:
        assert is_tree(branch)
    return [label] + list(branches)

# selector
def label(tree):
    return tree[0]

def branches(tree):
    return tree[1:]

def is_tree(tree):
    if type(tree) != list or len(tree) < 1:
        return False
    for branch in branches(tree): # each of the branch is also a tree
        if not is_tree(branch):
            return False
    return True

# whether a tree is leaf
def is_leaf(tree):
    return not branches(tree)

# fibonaci tree
def fib_tree(n):
    if n<= 1:
        return tree(n)
    else:
        left, right = fib_tree(n-2), fib_tree(n-1)
        return tree(label(left)+label(right), [left,right])

# count trees    
def count_leaves(t):
    if is_leaf(t):
        return 1
    else:
        return sum([count_leaves(b) for b in branches(t)])

# return a list containing leaf labels of tree
def leaves(tree):
    if is_leaf(tree):
        return [label(tree)]
    else:
        return sum([leaves(b) for b in branches(tree)], [])

# return a tree from another tree but with leaf labels incremented
def increment_leaves(t):
    if is_leaf(t):
        return tree(label(t) + 1)
    else:
        bs = [increment_leaves(b) for b in branches(t)]
        return tree(label(t), bs)
    
# return a tree like old tree with all labels incremented
def increment(t):
    return tree(label(t) + 1, [increment(b) for b in branches(t)])

# print a tree
def print_tree_b(t):
    print(label(t))
    for b in branches(t):
        print_tree_b(b)

def print_tree(t, indent = 0):
    print('  ' * indent + str(label(t)))
    for b in branches(t):
        print_tree(b, indent + 1)

def print_sums(t, so_far):
    so_far += label(t)
    if is_leaf(t):
        print(so_far)
    else: 
        for b in branches(t):
            print_sums(b, so_far)

numbers = tree(3, [tree(4), tree(5, [tree(6)])])
haste = tree('h', [tree('a', [tree('s'),tree('t')]), tree('e')])


def count_paths(t, total):
    if total == label(t):
        found = 1
    else:
        found = 0
    return found + sum([count_paths(b, total - label(t)) for b in branches(t)])

t = tree(3, [tree(-1), tree(1, [tree(2, [tree(1)]), tree(3)]), tree(1, [tree(-1)])])

# list
suits = ['coin', 'string', 'myriad']
original_suits = suits
suits.pop()
suits.remove('string')
suits.append('cup')
suits.extend(['sword','club'])
suits[2] = 'spade'
suits[0:2] = ['heart', 'diamond']
suits
original_suits

# disctorionaries

numerals = {'I': 1, 'V': 5, 'X': 10}
numerals['X']
numerals['X'] = 11
numerals['L'] = 50
numerals['L']
numerals.pop('X')
numerals.get('X')


four = [1,2,3,4]
len(four)
# mystery(four)

def mystery(s):
    s.pop()
    s.pop()

def mystery(s):
    s[2:] = []

# another_mystery()

def another_mystery(s):
    four.pop()
    four.pop()


# tuple
(3,4,5,6)
tuple()
tuple([3,4,5])
(2,) # a tuple containing 2
(3,4) + (5,6)
5 in (3,4,5)
{(1,2): 3} # tuples can be keys of a dictionary
# {[1,2]: 3} # not allowed to user a list as key in a dictionary

x = [1,2]
x+x

x.append(3)
x+x

s = ([1,2], 3)
s[0][0] = 4

def make_withdraw_list(balance):
    b = [balance]
    def withdraw(amount):
        if amount > b[0]:
            return 'Insufficient funds'
        b[0] = b[0] - amount
        return b[0]
    return withdraw

withdraw = make_withdraw_list(100)
withdraw(25)

# Trees

def tree(root_label, branches =[]):
    for branch in branches:
        assert is_tree(branch)
    return [root_label] + list(branches)

def label(tree):
    return tree[0]

def branches(tree):
    return tree[1:]

def is_tree(tree):
    if type(tree) != list or len(tree) < 1:
        return False
    for branch in branches(tree):
        if not is_tree(branch):
            return False
    return True

def is_leaf(tree):
    return not branches(tree)

def fib_tree(n):
    if n == 0 or n == 1:
        return tree(n)
    else:
        left, right = fib_tree(n-2), fib_tree(n-1)
        fib_n = label(left) + label(right)
        return tree(fib_n, [left, right])

# count leaf
def count_leaf(tree):
    if is_leaf(tree):
        return 1
    else:
        return sum([count_leaf(b) for b in branches(tree)])

def leaf(tree):
    if is_leaf(tree):
        return [label(tree)]
    else:
        return sum([leaf(b) for b in branches(tree)], [])


def incremental_leaf(t):
    if is_leaf(t):
        return tree(label(t) + 1)
    else:
        return tree(label(t), [incremental_leaf(b) for b in branches(t)])

def increment(t):
    return tree(label(t) + 1, [increment(b) for b in branches(t)])

def print_sums(t, so_far):
    so_far = so_far + label(t)
    if is_leaf(t):
        print(so_far)
    else:
        for b in branches(t):
            print_sums(b, so_far)


# iterators

def double(x):
    print("xx", x, "=>", 2*x, '**')
    return 2*x

def palindrome(s):
    return list(s) == list(reversed(s))

def palindrome_2(s):
    return all([a == b for a, b in zip(s, reversed(s))])


def evens(start, end):
    even = start + (start % 2)
    while even < end:
        yield even
        even += 2

def countdown(k):
    if k > 0:
        yield k
        yield from countdown(k-1)
    else:
        yield 'Blast off'

# Iterator and Generator
def prefixes(s):
    if s:
        yield from prefixes(s[:-1])
        yield s

def substrings(s):
    if s:
        yield from prefixes(s)
        yield from substrings(s[1:])


# return count of ways to sum to n with max m
def count_partitions(n, m):
    if n == 0:
        return 1
    elif n < 0 or m == 0:
        return 0
    else:
        with_m = count_partitions(n-m, m)
        without_m = count_partitions(n, m-1)
        return with_m + without_m


def count_partitions_n(n, m):
    if n < 0 or m == 0:
        return 0
    else:
        exact_match = 0
        if n == m:
            exact_match = 1
        with_m = count_partitions_n(n-m, m)
        without_m = count_partitions_n(n, m-1)
        return exact_match + with_m + without_m

# return list partitions to sum to n with max m
def list_partitions_n(n, m):
    if n < 0 or m == 0:
        return []
    else:
        exact_match = []
        if n == m:
            exact_match = [[m]]
        with_m = [element + [m] for element in list_partitions_n(n-m, m)]
        without_m = list_partitions_n(n, m-1)
        return exact_match + with_m + without_m


for p in list_partitions_n(6, 4):
    print(p)


def list_partitions_new(n, m):
    if n < 0 or m == 0:
        return []
    else:
        exact_match = []
        if n == m:
            exact_match = [str(m)] # list of string
        with_m = [element + ' + ' + str(m) for element in list_partitions_new(n-m, m)]
        without_m = list_partitions_new(n, m-1)
        return exact_match + with_m + without_m

for p in list_partitions_new(6, 4):
    print(p)


def partitions(n, m):
    if n < 0 or m == 0:
        return []
    else:
        exact_match = []
        if n == m:
            exact_match = [str(m)] # list of string
        with_m = [element + ' + ' + str(m) for element in list_partitions_new(n-m, m)]
        without_m = list_partitions_new(n, m-1)
        return exact_match + with_m + without_m


# Yield partitions
def partitions(n, m):
    if n > 0 and m > 0:
        if n == m:
            yield str(m)
        for p in partitions(n-m, m):
            yield p + ' + ' + str(m)
        yield from list_partitions_new(n, m-1)

# if want 10 examples of the sum above:
# for _ in range(10):
#     print(next(t))


def a_then_b(a,b):
    for x in a:
        yield x
    for x in b:
        yield x

def a_then_b_2(a,b):
    yield from a
    yield from b

list(a_then_b_2([3,4], [5,6]))


def countdown_1(k):
    if k > 0:
        yield k
        for x in countdown_1(k - 1):
            yield x


def countdown_2(k):
    if k > 0:
        yield k
        yield from countdown_2(k - 1)
    else:
        yield 'Blast off'


def prefixes(s):
    if s:
        yield from prefixes(s[:-1])
        yield s

def substrings(s):
    if s:
        yield from prefixes(s)
        yield from substrings(s[1:])


# Object oriented programming

class Account:
    
    interest = 0.02 # a class attribute

    def __init__(self, account_holder): 
        self.holder = account_holder # instance attribute
        self.balance = 0

    def deposit(self,amount):
        self.balance = self.balance + amount
        return self.balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            return 'Insufficient funds'
        self.balance = self.balance - amount
        return self.balance

# Class attributes
class Transaction:
    log = []

    def __init__(self, amount):
        self.amount = amount
        self.prior = list(self.log)
        self.log.append(amount)
    
    def balance(self):
        return self.amount + sum([t.amount for t in self.prior])
        

# Inheritance
class CheckingAccount(Account): # CheckingAccount is a specific type of Account and inherits from basic case - Account
    withdraw_fee = 1
    interest = 0.01
    def withdraw(self, amount):
        return Account.withdraw(self, amount + self.withdraw_fee)

class SavingsAccount(Account):
    deposit_fee = 2
    def deposit(self, amount):
        return Account.deposit(self, amount - self.deposit_fee)

# Multiple inheritance:
# Create a new account:
# Low interest rate of 1%; (from CheckingAccount from Account)
# A $1 fee for withdrawals; (from CheckingAccount)
# A $2 fee for deposits; (from SavingsAccount)
# A free dollar when you open your account
class AsSeenOnTVAccount(CheckingAccount, SavingsAccount):
    def __init__(self, account_holder):
        self.holder = account_holder
        self.balance = 1 # start with a free dollar


# Composition
class Bank:
    """ A bank has accounts.
    >>> bank = Bank()
    >>> john = bank.open_account('John', 10)
    >>> jack = bank.open_account('Jack', 5, CheckingAccount)
    >>> john.interest
    0.02
    >>> jack.interest
    0.01
    >>> bank.pay_interest()
    >>> john.balance
    10.2
    >>> bank.too_big_to_fail()
    True
    """

    def __init__(self):
        self.accounts = []
    
    def open_account(self, holder, amount, account_type = Account):
        account = account_type(holder) # open account
        account.deposit(amount) # deposit money into account
        self.accounts.append(account) # put created account into our accounts in the bank
        return account
    
    def pay_interest(self):
        for i in self.accounts: # for each accounts, the bank pays interest on them
            i.deposit(i.balance * i.interest)
    
    def too_big_to_fail(self):
        return len(self.accounts) > 1 # The bank cannot contain more than 1 accounts

    

class A:
    x, y, z = 0, 1, 2
    def f(self):
        return [self.x, self.y, self.z]

class B(A):
    x = 6
    def __init__(self):
        self.z = 'A'

class Ratio:
    def __init__(self, n, d):
        self.numer = n
        self.denom = d
    
    def __repr__(self):
        return 'Ratio({0}, {1})'.format(self.numer, self.denom)
    
    def __str__(self):
        return '{0}/{1}'.format(self.numer, self.denom)
    
    def __add__(self, other): # other passed in as an object
        if isinstance(other, int): # Type Dispatching
            n = self.numer + self.denom * other
            d = self.denom
        elif isinstance(other, Ratio):
            n = self.numer * other.denom +  self.denom * other.numer
            d = self.denom * other.denom
        elif isinstance(other, float):
            return float(self) + other # Type Coercion
        g = gcd(n,d)
        return Ratio(n//g, d//g)

    __radd__ = __add__

    def __float__(self):
        return self.numer/self.denom

def gcd(n,d):
    while n != d:
        n,d = min(n,d), abs(n-d)
    return n



# linked list

class Link:   
    empty = () # empty tuple

    def __init__(self, first, rest = empty):
        assert rest is Link.empty or isinstance(rest, Link)
        self.first = first
        self.rest = rest
    
    def __repr__(self):
        if self.rest:
            rest_repr = ', ' + repr(self.rest)
        else:
            rest_repr = ''
        return 'Link(' + repr(self.first) + rest_repr + ')'
    
    def __str__(self):
        string = '<'
        while self.rest is not Link.empty:
            string += str(self.first) + ' '
            self = self.rest
        return string + str(self.first) + '>'

# help(isinstance): return whether an object is an instance of a class or of a subclass thereof.
# whether rest has type of link; and also true if rest is an instance of a class that inherits from link
# if we want to build a special kind of linked list by inheriting from link. we can still use the same constructor

s = Link(3, Link(4, Link(5)))
s.first
s.rest
s.rest.first
s.rest.rest.first
s.rest.rest.rest is Link.empty
s.rest.first = 7 # change value to be 7
Link(8, s.rest) # create a new link that starts with 8 and rest of old list



# Linked list processing
square, odd = lambda x: x * x, lambda x: x % 2 == 1
list(map(square, filter(odd, range(1,6))))


def range_link(start, end):
    """ Return a Link containing consecutive integers from start to end.
    >>> range_link(3, 6)
    Link(3, Link(4, Link(5)))
    """
    if start >= end:
        return Link.empty
    else:
        return Link(start, range_link(start + 1, end))

def map_link(f, s):
    """Return a Link that contains f(x) for each x in Link s.
    >>> map_link(square, range_link(3, 6))
    Link(9, Link(16, Link(25)))
    """
    if s is Link.empty:
        return s
    else:
        return Link(f(s.first), map_link(f, s.rest))

def filter_link(f, s):
    """Return a Link that contains only the elements x of Link s for which f(x) is a true value.
    >>> filter_link(odd, range_link(3, 6))
    Link(3, Link(5))
    """
    if s is Link.empty:
        return s
    elif f(s.first):
        return Link(s.first, filter_link(f, s.rest))
    else:
        return filter_link(f, s.rest)

# map_link(square, filter_link(odd, range_link(1,6)))

# Adding to a set represented as an ordered list
def add(s, v):
    """ Add v to s, returning modified s.

    >>> s = Link(1, Link(3, Link(5)))
    >>> add(s, 0)
    Link(0, Link(1, Link(3, Link(5))))
    >>> add(s, 3)
    Link(0, Link(1, Link(3, Link(5))))
    >>> add(s, 4)
    Link(0, Link(1, Link(3, Link(4, Link(5)))))
    >>> add(s, 6)
    Link(0, Link(1, Link(3, Link(4, Link(5, Link(6))))))    
    """

    assert s is not Link.empty
    if s.first > v:
        s.first, s.rest = v, Link(s.first, s.rest)
    elif s.first < v and s.rest is Link.empty:
        s.rest = Link(v) # v is the last link instance, rest is empty
    elif s.first < v:
        add(s.rest, v) # keep searching in s.rest for v
    return s


# Slicing a linked list

def slice_link(s, i, j):
    """Return a linked list containing elements from i:j.
    >>> evens = Link(4, Link(2, Link(6)))
    >>> slice_link(evens, 1, 100)
    Link(2, Link(6))
    >>> slice_link(evens, 1, 2)
    Link(2)
    >>> slice_link(evens, 0, 2)
    Link(4, Link(2))
    >>> slice_link(evens, 1, 1) is Link.empty
    """

    assert i >= 0 and j >= 0
    if j == 0 or s is Link.empty:
        return Link.empty
    elif i == 0:
        return Link(s.first, slice_link(s.rest, i, j - 1))
    else:
        return slice_link(s.rest, i - 1, j - 1)



# Inserting into a Linked List

def insert_link(s, x, i):
    """ Insert x into linked list s at index i.
    >>> evens = Link(4, Link(2, Link(6)))
    >>> insert_link(evens, 8, 1)
    Link(4, Link(8, Link(2, Link(6))))
    >>> insert_link(evens, 10, 4)
    Link(4, Link(8, Link(2, Link(6, Link(10)))))
    >>> insert_link(evens, 12, 0)
    Link(12, Link(4, Link(8, Link(2, Link(6, Link(10))))))
    >>> insert_link(evens, 14, 10)
    Index out of range
    >>> print(evens)
    <12 4 8 2 6 10> 
    """
    print(s, x, i)
    if s is Link.empty:
        print('Index out of range')
    elif i == 0: # insert into first position
        s.first, s.rest = x, Link(s.first, s.rest)
    elif i == 1 and s.rest is Link.empty: # insert into last position
        s.rest = Link(x)
    else:
        insert_link(s.rest, x, i - 1)
    return s

# prefix sum: a sequence of numbers is the sum of the first n elements for some positive length n
def tens(s):
    def f(suffix, total):
        if total % 10 == 0:
            print(total)
        if suffix is not Link.empty:
            f(suffix.rest, total + suffix.first)
    f(s.rest, s.first)


# Tree class

class Tree:
    def __init__(self, label, branches=[]):
        self.label = label
        for branch in branches:
            assert isinstance(branch, Tree)
        self.branches = list(branches)
    
    def __repr__(self):
        if self.branches:
            branch_str = ', ' + repr(self.branches)
        else:
            branch_str = ''
        return 'Tree({0}{1})'.format(repr(self.label), branch_str)
    
    def __str__(self):
        return '\n'.join(self.indented())
    
    def indented(self):
        lines = []
        for b in self.branches:
            for line in b.indented():
                lines.append('  ' + line)
        return [str(self.label)] + lines
    
    def is_leaf(self):
        return not self.branches

# define a febonacci tree
def fib_tree(n):
    if n == 0 or n == 1:
        return Tree(n)
    else:
        left = fib_tree(n-2)
        right = fib_tree(n-1)
        fib_n = left.label + right.label
        return Tree(fib_n, [left, right])

# return a list of leaf labels in tree t
def leaves(t):
    if t.is_leaf():
        return [t.label]
    else:
        all_leaves = []
        for b in t.branches:
            all_leaves.extend(leaves(b)) # extend will flatten all the lists of leaves; append will leave all list out
        return all_leaves

# return the number of transactions in the longest path in T
def height(t):
    if t.is_leaf():
        return 0
    else:
        return 1 + max([height(b) for b in t.branches])


# Pruning trees
def prune(t, n):
    """ Prune all subtrees whose label is n."""
    t.branches = [b for b in t.branches if b.label != n]
    for b in t.branches:
        prune(b, n)

# efficiency
def fib(n):
    if n == 0 or n == 1:
        return n
    else:
        return fib(n-2) + fib(n-1)


def count(f):
    def counted(n):
        counted.call_count += 1
        return f(n)
    counted.call_count = 0
    return counted

def memo(f):
    cache = {}
    def memorized(n):
        if n not in cache:
            cache[n] = f(n)
        return cache[n]
    return memoized 


# Restaurants search

def search(query, ranking = lambda r: -r.stars):
    results = [r for r in Restaurant.all if query in r.name]
    return sorted(results, key = ranking)

# def reviewed_both(r, s):
#     return len([x for x in r.reviewers if x in s.reviewers])

def reviewed_both(r, s):
    return fast_overlap(r.reviewers, s.reviewers)

def fast_overlap(s, t):
    count, i, j = 0, 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            count, i, j = count + 1, i + 1, j + 1
        elif s[i] < t[j]:
            i += 1
        else: 
            j += 1
        return count



class Restaurant:
    all = []
    def __init__(self, name, stars, reviewers):
        self.name, self.stars = name, stars
        self.reviewers = reviewers
        Restaurant.all.append(self)
    
    def similar(self, k, similarity = reviewed_both):
        "Return the K most simimlar restaurants to SELF."
        others = list(Restaurant.all)
        others.remove(self)
        return sorted(others, key = lambda r: -similarity(self, r))[:k]
    
    def __repr__(self):
        return '<' + self.name + '>'

# import json
# reviewers_for_restaurant = {}
# for line in open('reviews.json'):
#     r = json.loads(line)
#     biz = r['business_id']
#     if biz not in reviewers_for_restaurant:
#         reviewers_for_restaurant[biz] = [r['user_id']]
#     else:
#         reviewers_for_restaurant[biz].append(r['user_id'])

# for line in open('restaurants.json'):
#     r = json.loads(line)
#     reviewers = reviewers_for_restaurant[r['business_id']]
#     Restaurant(r['name'], r['stars'], reviewers)

while True:
    print('>', end = ' ')
    results = search(input().strip())
    for r in results:
        print(r, 'shares reviewers with', r.similar(3))

Restaurant('Thai Delight', 2)
Restaurant('Thai Basil', 2)
Restaurant('Top Dog', 5)

results = search('Thai')
for r in results:
    print(r, 'is similar to', r.similar(3))


# Append a value to a linked list
# Recursion
def append(s, x):
    if s.rest is not Link.empty:
        append(s.rest, x)
    else:
        s.rest = Link(x)

# Iteration
def append(s, x):
    while s.rest is not Link.empty:
        s = s.rest
    s.rest = Link(x)

# Pop
def pop(s, i):
    assert i > 0 and i < length(s)
    for x in range(i-1):
        s = s.rest
    result = s.rest.first
    s.rest = s.rest.rest
    return result


def ordered(s, key = lambda x: x):
    """IS Link s ordered?
    >>> ordered(Link(1, Link(3, Link(4))))
    True
    >>> ordered(Link(1, Link(4, Link(3))))
    False
    >>> ordered(Link(1, Link(-3, Link(4))))
    False
    >>> ordered(Link(1, Link(-3, Link(4))), key = abs)
    True
    >>> ordered(Link(1, Link(4, Link(3))), key = abs)
    """

    if s is Link.empty or s.rest is Link.empty:
        return True
    elif key(s.first) > key(s.rest.first):
        return False
    else:
        return ordered(s.rest)


def merge(s, t):
    """
    >>> a = Link(1, Link(5))
    >>> b = Link(1, Link(4))
    >>> merge(a, b)
    Link(1, Link(1, Link(4, Link(5))))
    """

    if s is Link.empty:
        return t 
    elif t is Link.empty:
        return s
    elif s.first <= t.first:
        return Link(s.first, merge(s.rest, t))
    else:
        return Link(t.first, merge(s, t.rest))


def merge_in_place(s, t):
    if s is Link.empty:
        return t 
    elif t is Link.empty:
        return s
    elif s.first <= t.first:
        s.rest = merge_in_place(s.rest, t)
        return s
    else:
        t.rest = merge_in_place(s, t.rest)
        return t


def largest_adj_sum(s):
    return max([s[i] + s[i+1] for i in range(len(s) - 1)])


def digit_dict(s):
    return {d: [x for x in s if x % 10 ==d] for d in range(10) if any([x % 10 == d for x in s])}


def all_have_an_equal(s):
    return all([s[i] in s[:i] + s[i+1:] for i in range(len(s))])

def leaves(t):
    if t.is_leaf():
        return [t.label]
    else:
        all_leaves = []
        for b in t.branches:
            all_leaves.extend(leaves(b))
        return all_leaves

def height(t):
    if t.is_leaf():
        return 0
    else:
        return 1 + max([height(b) for b in t.branches])


def prune(t, n):
    t.branches = [b for b in t.branches if b.label != n]
    for b in t.branches:
        prune(b, n)

