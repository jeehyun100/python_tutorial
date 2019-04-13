
from students import Students
from collections import MutableSequence

class StudentCreditsList(MutableSequence):
    """ 학점관리 리스트 클래스
    Students object를 관리해주는 클래스
    """

    def __init__(self, data=None):
        """Initialize the class"""
        super(StudentCreditsList, self).__init__()
        if (data is not None):
            self._list = list(data)
        else:
            self._list = list()

    def __repr__(self):
        return '\n'.join(str(list_v) for list_v in self._list)

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
        first_line_for_column = '\t'.join(str(_v).ljust(8, " ") for _v in self._list[0].columns()) + '\n'
        return first_line_for_column + '\n'.join(str(list_val) for list_val in self._list)

    def __eq__(self, other):
        return self._list == other

    def __ne__(self, other):
        return self._list != other

    def sort(self, key=None, reverse=False):
        self._list.sort(key=key, reverse=reverse)

    def insert(self, ii, val):
        # optional: self._acl_check(val)
        self._list.insert(ii, val)

    def remove(self, ii):
        # optional: self._acl_check(val)
        self.__delitem__(ii)

    def append(self, val):
        val._index = len(self._list)+1
        self.insert(len(self._list), val)

    def save(self, files):
        with open("./"+files, 'w') as datafile:
            datafile.write(self.__repr__())

if __name__=='__main__':
    with open("1.txt") as f:
        lines_all = f.readlines()
    foo = [ Students(line.replace('\n','').split('\t')) for line in lines_all]
    voo = StudentCreditsList(foo)
    for list_val in voo:
        list_val.mean = (list_val._midterm + list_val._finalterm)/2
    dd = voo.sort( key=lambda x : (x._mean, x._name), reverse=True)
    print(voo)
