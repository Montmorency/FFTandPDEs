import copy

class Foo(object):
  def __init__(self, val):
    self.val = val

  def __repr__(self):
    return str(self.val)

foo = Foo(1)
print foo
a = ['foo',foo]
print a
