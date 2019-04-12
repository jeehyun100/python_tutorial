import collections

# class TypedList(collections.MutableSequence):
#
#     def __init__(self, oktypes, *args):
#         self.oktypes = oktypes
#         self.list = list()
#         self.extend(list(args))
#
#     def check(self, v):
#         if not isinstance(v, self.oktypes):
#             raise TypeError, v
#
#     def __len__(self): return len(self.list)
#
#     def __getitem__(self, i): return self.list[i]
#
#     def __delitem__(self, i): del self.list[i]
#
#     def __setitem__(self, i, v):
#         self.check(v)
#         self.list[i] = v
#
#     def insert(self, i, v):
#         self.check(v)
#         self.list.insert(i, v)
#
#     def __str__(self):
#         return str(self.list)
#

from collections import MutableSequence

class StudentCreditsList(MutableSequence):
    """A container for manipulating lists of hosts"""
    def __init__(self, data=None):
        """Initialize the class"""
        super(StudentCreditsList, self).__init__()
        if (data is not None):
            self._list = list(data)
        else:
            self._list = list()

    def __repr__(self):
        return "<{0} {1}>".format(self.__class__.__name__, self._list)

    def __len__(self):
        """List length"""
        return len(self._list)

    def __getitem__(self, ii):
        """Get a list item"""
        return self._list[ii]

    def __delitem__(self, ii):
        """Delete an item"""
        del self._list[ii]

    def __setitem__(self, ii, val):
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __str__(self):
        return str(self._list)

    def insert(self, ii, val):
        # optional: self._acl_check(val)
        self._list.insert(ii, val)

    def append(self, val):
        self.insert(len(self._list), val)

if __name__=='__main__':

    with open("data.txt") as f:
        lines_all = f.readlines()

    #
    #
    # foo = StudentCreditsList([{'id':1,"s_id":1998, "name" : "Paik"},
    #                           {'id': 1, "s_id": 1998, "name": "Paik"}])
    #foo.append(6)
    print (foo)  # <MyList [1, 2, 3, 4, 5, 6]>