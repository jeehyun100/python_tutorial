from students import Students
from collections import MutableSequence

class StudentCreditsList(MutableSequence):
    """학점 관리 기본 클래스 asasd
    # StudentCreitsList 클래스와 Students Import하여
    불러오기, 등록, 삭제의 기능을 수행한다.
    """

    def __init__(self, data=None):
        """
            Need to put comments
        """
        super(StudentCreditsList, self).__init__()
        if (data is not None):
            self._list = list(data)
        else:
            self._list = list()

    def __repr__(self):
        return '\n'.join(str(list_v) for list_v in self._list)

    def __len__(self):
        """
            Need to put comments
        """
        return len(self._list)

    def __getitem__(self, ii):
        """
            Need to put comments
        """
        return self._list[ii]

    def __delitem__(self, ii):
        """
            Need to put comments
        """
        del self._list[ii]

    def __setitem__(self, ii, val):
        """
            Need to put comments
        """
        # optional: self._acl_check(val)
        self._list[ii] = val

    def __str__(self):
        """
            Nedd to put comments
        """
        first_line_for_column = '\t'.join(str(_v).ljust(8, " ") for _v in self._list[0].columns()) + '\n'
        return first_line_for_column + '\n'.join(str(list_val) for list_val in self._list)

    def __eq__(self, other):
        """
            Nedd to put comments
        """
        return self._list == other

    def __ne__(self, other):
        """
            Nedd to put comments
        """
        return self._list != other

    def sort(self, key=None, reverse=False):
        """
            Nedd to put comments
        """
        self._list.sort(key=key, reverse=reverse)

    def insert(self, ii, val):
        """
            Nedd to put comments
        """
        pass

    def remove(self, ii):
        """
            Nedd to put comments
        """
        pass

    def append(self, val):
        """
            Nedd to put comments
        """
        pass

    def save(self, files):
        """
            Nedd to put comments
        """
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
