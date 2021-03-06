from students import Students
from collections import MutableSequence
import unicodedata


class StudentCreditsList(MutableSequence):
    """ Manage for Student Credits List
    Student credit list inherited the Mutable Sequence have function
    add, delete, modify, etc.

    """

    def __init__(self, data=None):
        """Creates a "StudentCreditsList"
           Make a list from datafile

        """
        super().__init__()
        if data is not None:
            self._list = list(data)
        else:
            self._list = list()

    def __repr__(self):
        """String representation of an StudentCreditsList class

        return:
              String,

        """
        # for list_v in self._list:
        #     print(list_v)
        return '\n'.join(repr(list_v) for list_v in self._list)

    def __len__(self):
        """The Length of list

        return:
              integer,
        """
        return len(self._list)

    def __getitem__(self, ii):
        """Get elements from a list.

        return:
              The element of list,
        """
        return self._list[ii]

    def __delitem__(self, ii):
        """Delete elements from a list.

        """
        del self._list[ii]

    def __setitem__(self, ii, val):
        """Set elements to a list.

        """
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __str__(self):
        """String representation of a StudentCreditsList class
        The first line is columns.

        return:
              String,

        """
        _cols = [_v for _v in self._list[0].columns()]
        first_line_for_column = "{:<4s} {:<6s} {:<11s} {:<8s} {:<4s} {:<4s} {:<6s} {:<6s}\n"\
            .format(_cols[0], _cols[1], _cols[2], _cols[3], _cols[4], _cols[5], _cols[6], _cols[7])
        return first_line_for_column + '\n'.join(str(_item) for _item in self._list)

    def sort(self, key=None, reverse=False):
        """Sort a studentCreditsList class

        return:
              List,

        """
        self._list.sort(key=key, reverse=reverse)

    def insert(self, ii, val):
        """Do not use it
        """
        pass

    def remove(self, ii):
        """remove item in the list
        """
        self._list.remove(ii)

    def append(self, val):
        """Append item into the list
        """
        self._list.append(val)

    def save(self, files):
        """ Save a studentCreditsList class
        """
        with open("./"+files, 'w', encoding='utf-8') as datafile:
            datafile.write(self.__repr__())


if __name__ == '__main__':
    with open("data.txt", encoding='utf-8') as f:
        lines_all = f.readlines()
    foo = [Students(line.split()) for line in lines_all]
    voo = StudentCreditsList(foo)
    print(voo)
    voo.sort(key=lambda x: (x.mean, x.name), reverse=True)
    print(voo)
    voo.append(Students((16, '11111111', 'e', '2000-01-01', 30, 34)))
    print(voo)
    students =  (
        filter(lambda item: item._id == '11111111', voo)
    )
    for student in students:
        voo.remove(student)
    print(voo)
