import os
import sys
import numpy as np
import pandas as pd
from grade_management import GradeManagement


class GradeManagementPandas(GradeManagement):

    def __init__(self):
        self.columns = ['id', 'name', 'birthday', 'midterm', 'finalterm', 'average', 'grade']
        self.dtype = {
            'id': 'object',
            'name': 'object',
            'birthday': 'datetime64',
            'midterm': 'int64',
            'finalterm': 'int64',
            'average': 'float64',
            'grade': 'object'
        }

        self.student_list = pd.DataFrame(columns=self.columns)
        self.student_list = self.student_list.astype(self.dtype)
        self.tabs = '  '
        self._filename = None

    def merge_list(self, new_list):
        self.student_list = self.student_list.append(new_list, sort=False, ignore_index=True)
        self.student_list.index = self.student_list.index + 1
        self.student_list.average = self.student_list[['midterm', 'finalterm']].mean(axis=1)
        self.student_list.grade = self.student_list.average.apply(self.calc_grade)

    def calc_grade(self, average):
        if 95 <= average:
            return 'S'
        elif 90 <= average:
            return 'A'
        elif 80 <= average:
            return 'B'
        elif 70 <= average:
            return 'C'
        elif 60 <= average:
            return 'D'
        else:
            return 'F'

    def find_student(self):
        opt = self.input_options(['id', 'name'], 1, 'How do you want to find the student?')
        if opt.upper() == 'ID':
            id = self.input_id(1, "Input ID of Student You're Looking for")
            return self.student_list[self.student_list.id == id]
        else:
            name = self.input_name(1, "Input Name of Student You're Looking for")
            return self.student_list[self.student_list.name == name]

    def add_a_new_entry(self):
        id = self.input_id()
        name = self.input_name()
        birthday = self.input_birthday()
        midterm = self.input_score(1, 'Input Midterm Score')
        finalterm = self.input_score(1, 'Input Finalterm Score')

        new_list = pd.DataFrame(
            [[id, name, pd.Timestamp(birthday), midterm, finalterm, np.nan, np.nan]],
            columns=self.columns)
        new_list.astype(self.dtype)

        self.merge_list(new_list)

    def delete_an_entry(self):
        target_list = self.find_student()

        if len(target_list):
            print('You selected the list below.')
            print(target_list)
            opt = self.input_options(['y', 'n'], 1, 'Do you really want to delete?')

            if opt.upper() == 'Y':
                self.student_list.drop(target_list.index, inplace=True)

                if len(self.student_list):
                    self.student_list.index = range(len(self.student_list))
                    self.student_list.index = self.student_list.index + 1

    def find_some_item_from_entry(self):
        target_list = self.find_student()

        print('{:10s}{:10s}{:10s}'.format('일련번호', '평균', 'Grade'))
        print(target_list[['average', 'grade']].to_string(header=False, col_space=10))

    def modify_an_entry(self):
        target_list = self.find_student()

        opt = self.input_options(['midterm', 'finalterm'], 1, 'Which test do you want to modify?')
        score = self.input_score()

        if opt.upper() == 'MIDTERM':
            self.student_list[self.student_list.index == target_list.index].midterm = score
        else:
            self.student_list[self.student_list.index == target_list.index].finalterm = score

    def print_dataframe(self, df):
        header = [
                '일련번호',
                '학생 id',
                '이름',
                '생년월일',
                '중간고사',
                '기말고사',
                '평균',
                'Grade'
            ]

        header_str = '{:10s}' * len(header)
        print(header_str.format(*header))
        print(df.to_string(header=False, col_space=10))

    def print_the_contents_of_all_entries(self):

        if len(self.student_list):
            self.print_dataframe(self.student_list)
        else:
            print('There is no contents to show')

    def read_personal_data(self):
        self._filename = self.input_filename()
        try:
            new_list = pd.read_csv(
                self._filename,
                sep="\s+",
                names=['index'] + self.columns,
                index_col=['index'],
                parse_dates=['birthday'],
                dtype={'id':'object', 'grade':'object'}
            )

            self.merge_list(new_list)
        except pd.errors.EmptyDataError:
            print('The file is empty.')

    def sort_entries(self):
        if not len(self.student_list):
            print('There is no contents to sort')
            return

        opt = self.input_options(['n', 'a', 'g'], 1, 'Sort by name(n) or average(a) or grade(g)')
        if opt.upper() == 'N':
            self.print_dataframe(self.student_list.sort_values(by=['name', 'average'], ascending=[True,False]))
        elif opt.upper() == 'A' or opt.upper() == 'G':
            self.print_dataframe(self.student_list.sort_values(by=['average', 'name'], ascending=[False,True]))

    def write_the_contents_to_the_same_file(self):
        """The function to save all student credits

        """
        if not len(self.student_list):
            print('There is no contents to write')
            return

        if self._filename is None:
            self._filename = self.input_filename()

        with open(self._filename, 'w') as OUT:
            OUT.write(self.student_list.to_string(header=False, index_names=False))
        print(f'Data are saved into {self._filename!r}')

if __name__ == '__main__':

    GradeManagementPandas().run()
