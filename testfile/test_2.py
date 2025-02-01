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

def min_abs_indices(s):
    min_abs = min(map(abs, s))
    return [i for i in range(len(s)) if abs(s[i]) == min_abs]

def largest_adj_sum(s):
    return max([s[i] + s[i+1] for i in range(len(s) - 1)])
    # list(zip(s[:-1], s[1:]))
    # max([a+b for a,b in zip(s[:-1], s[1:])])


def digit_dict(s):
    return {d: [x for x in s if x % 10 ==d] for d in range(10) if any([x % 10 == d for x in s])}
    # Need to filter out digits that dont have any count ie. 0, 2 etc
    # {0: [], 1: [21], 2: [], 3: [13], 4: [34], 5: [5, 55], 6: [], 7: [], 8: [8], 9: [89]}


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
    return t


class Tree:
    """
    >>> t = Tree(3, [Tree(2, [Tree(5)]), Tree(4)])
    >>> t.label
    3
    >>> t.branches[0].label
    2
    >>> t.branches[1].is_leaf()
    True
    """
    def __init__(self, label, branches=[]):
        for b in branches:
            assert isinstance(b, Tree)
        self.label = label
        self.branches = list(branches)

    def is_leaf(self):
        return not self.branches

