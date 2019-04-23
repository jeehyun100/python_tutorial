import os
import sys
import numpy as np
import pandas as pd
from grade_management import GradeManagement


class GradeManagementPandas(GradeManagement):
    """
    Manage the student's credits with pandas library

    The "GradeManagementPandas" class inherits from the "GradeManagement" class
    """

    def __init__(self):
        self.columns = ['id', 'name', 'birthday', 'midterm', 'finalterm', 'average', 'grade']
        self.columns_to_save = ['id', 'name', 'birthday', 'midterm', 'finalterm']
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
        """
        Merge new list into the student list which is in the from of pandas dataframe
        """
        self.student_list = self.student_list.append(new_list, sort=False, ignore_index=True)
        self.student_list.index = self.student_list.index + 1
        self.student_list.average = self.student_list[['midterm', 'finalterm']].mean(axis=1)
        self.student_list.grade = self.student_list.average.apply(self.calc_grade)

    def calc_grade(self, average):
        """
        Calculate a grade from the average score
        """
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
        """
        Find student info from the student list.
        User specifies by which mean they will find studendt.
        It would be by ID or NAME.
        """
        opt = self.input_options(['id', 'name'], 1, 'How do you want to find the student?')
        if opt.upper() == 'ID':
            id = self.input_id(1, "Input ID of Student You're Looking for")
            return self.student_list[self.student_list.id == id]
        else:
            name = self.input_name(1, "Input Name of Student You're Looking for")
            return self.student_list[self.student_list.name == name]

    def add_a_new_entry(self):
        """
        Add student info into the student list.
        """
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
        """
        Delete student info from the student list.
        """
        target_list = self.find_student()

        if not len(target_list):
            print('There is no contents to show')
        else:
            print('You selected the list below.')
            self.print_dataframe(target_list)
            opt = self.input_options(['y', 'n'], 1, 'Do you really want to delete?')

            if opt.upper() == 'Y':
                self.student_list.drop(target_list.index, inplace=True)

                if len(self.student_list):
                    self.student_list.index = range(len(self.student_list))
                    self.student_list.index = self.student_list.index + 1

    def find_some_item_from_entry(self):
        """
        Find students and display into the screen.
        """
        target_list = self.find_student()

        if not len(target_list):
            print('There is no contents to show')
        else:
            print('{:10s}{:10s}{:10s}'.format('일련번호', '평균', 'Grade'))
            print(target_list[['average', 'grade']].to_string(header=False, col_space=10))

    def modify_an_entry(self):
        """
        Modify the midterm or fianlterm score.
        User specify which test score modify.
        """
        target_list = self.find_student()

        if not len(target_list):
            print('There is no contents to show')
        else:
            opt = self.input_options(['midterm', 'finalterm'], 1, 'Which test do you want to modify?')
            score = self.input_score()

            if opt.upper() == 'MIDTERM':
                for idx in target_list.index:
                    self.student_list.loc[self.student_list.index == idx, 'midterm'] = score
            else:
                for idx in target_list.index:
                    self.student_list.loc[self.student_list.index == idx, 'finalterm'] = score

    def print_dataframe(self, df):
        """
        Print Pandas dataframe with the specified header
        """
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
        """
        Print the student list into the screen
        """

        if len(self.student_list):
            self.print_dataframe(self.student_list)
        else:
            print('There is no contents to show')

    def read_personal_data(self):
        """
        Read data into the student list from the file user specified.
        """
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
        except pd.errors.EmptyDataError as e:
            print(f'The file is empty [{e!r}].')

    def sort_entries(self):
        """
        Sort the student list by the order user specified
        """
        if not len(self.student_list):
            print('There is no contents to sort')
            return

        opt = self.input_options(['n', 'a', 'g'], 1, 'Sort by name(n) or average(a) or grade(g)')
        if opt.upper() == 'N':
            self.print_dataframe(self.student_list.sort_values(by=['name', 'average'], ascending=[True,False]))
        elif opt.upper() == 'A' or opt.upper() == 'G':
            self.print_dataframe(self.student_list.sort_values(by=['average', 'name'], ascending=[False,True]))

    def write_the_contents_to_the_same_file(self):
        """
        Save the student list into the file that data had been read from.
        If the file is not specified, ask.
        """
        if not len(self.student_list):
            print('There is no contents to write')
            return

        if self._filename is None:
            self._filename = self.input_filename()

        with open(self._filename, 'w') as OUT:
            OUT.write(self.student_list.to_csv(date_format='%Y-%m-%d',
                sep='\t', header=False, columns=self.columns_to_save))
        print(f'Data are saved into {self._filename!r}')

if __name__ == '__main__':

    GradeManagementPandas().run()
