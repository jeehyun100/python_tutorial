# -*- coding: utf-8 -*-

import datetime


class Students:
    """ Manage the student's credits

    The "GradeManagement" class manage student's credits.
    This class have functions add, delete, find, modify, print
    , read, sort, quit, write files.

    """

    def __init__(self, *args):
        """Creates a "Students"
           Students class have properties which are index, id, name .. etc.
           The average(mid, final term) and grade are automatically calculated.

        Args:
              *args: list, students information from datafile.

        """
        if len(args) != 0:
            data = args[0]
            self._index = int(data[0])
            self._id = data[1]
            self._name = data[2]
            self._birthday = data[3]
            self._midterm = int(data[4])
            self._finalterm = int(data[5])
            self._mean = (self._midterm + self._finalterm)/2
            # Calculate grade automately
            self._grade = 'F' if self._mean <= 60 \
                else 'S' if (95 <= self._mean) \
                else 'A' if (90 <= self._mean) \
                else 'B' if (80 <= self._mean) \
                else 'C' if (70 <= self._mean) \
                else 'D'
        else:
            self._index = 0
            self._id = '0'
            self._name = ''
            self._birthday = ''
            self._midterm = 0
            self._finalterm = 0
            self._mean = 0
            self._grade = 'F'

    def __str__(self):
        """String representation of an students class

        return:
              String, split char '\t'

        """
        return u"{0} {1} {2}\t{3} {4} {5} {6} {7}".format(
            str(self._index).ljust(8, " "), str(self._id).ljust(8, " "),
            str(self._name).ljust(10, " "), str(self._birthday).ljust(12, " "),
            str(self._midterm).ljust(8, " "), str(self._finalterm).ljust(8, " "),
            str(self._mean).ljust(8, " "), str(self._grade).ljust(8, " "))

    def __repr__(self):
        """String representation of an students class for save

        return:
              String, split char '\t'

        """
        return u"{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
            str(self._index).ljust(8, " "), str(self._id).ljust(8, " "),
            str(self._name).ljust(10, " "), str(self._birthday).ljust(8, " "),
            str(self._midterm).ljust(8, " "), str(self._finalterm).ljust(8, " "),
            str('').ljust(8, " "), str('').ljust(8, " "))

    def cal_mean_grade(self):
        """Calculate grade

        """
        self._mean = (self._midterm + self._finalterm) / 2
        self._grade = 'F' if self._mean <= 60 \
            else 'S' if (95 <= self._mean) \
            else 'A' if (90 <= self._mean) \
            else 'B' if (80 <= self._mean) \
            else 'C' if (70 <= self._mean) \
            else 'D'


    @property
    def index(self):
        """ Student index

        return:
              Ingeger

        """
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def id(self):
        """ Student id

        return:
              String

        """
        return self._id

    @property
    def name(self):
        """ Student name

        return:
              String

        """
        return self._name

    @property
    def birthday(self):
        """ Student birthday

        return:
              String, format YYYY-MM-DD

        """
        return self._birthday

    @property
    def midterm(self):
        """ Student midterm

        return:
              Ingeger

        """
        return self._midterm

    @midterm.setter
    def midterm(self, value):
        try:
            self._midterm = int(value)
            self.cal_mean_grade()
        except ValueError as e:
            raise ValueError(str(e)+" for {0} column ".format('midterm'))

    @property
    def finalterm(self):
        """ Student final term

        return:
              Ingeger

        """
        return self._finalterm

    @finalterm.setter
    def finalterm(self, value):
        try:
            self._finalterm = int(value)
            self.cal_mean_grade()
        except ValueError as e:
            raise ValueError(str(e)+" for {0} column ".format('finalterm'))

    @property
    def mean(self):
        """ Student average midterm and final term

        return:
              float

        """
        return self._mean

    @property
    def grade(self):
        """ Student grade base on average

        return:
              String

        """
        return self._grade

    @classmethod
    def columns(cls):
        """Show columns infomation

        return:
            Dictionary, {column name : display flag, ...}

        """
        # 입력 받아야 할 컬럼과 보여주는 컬럼을 설정함
        return {"일련번호": False, "학생 id": True, "이름": True, "생년월일": True,
                "중간고사": True, "기말고사": True, "평균": False, "Grade": False}
