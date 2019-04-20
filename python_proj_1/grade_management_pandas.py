import os
import sys
import datetime
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

    def input_(self, level=0, input_description=''):
        tabs = self.tabs * level
        return input(tabs + input_description)

    def input_id(self, level=1, input_description='Input ID'):
        while True:
            id = self.input_(
                level,
                input_description + ' (format: XXXXXXXX): ')

            if len(id) == 6:
                print(self.tabs * (level+1), f'Your Input: {id!r}')
                return id

    def input_name(self, level=1, input_description='Input Name'):
        name = self.input_(level, input_description + ' : ')
        print(self.tabs * (level+1), f'Your Input: {name!r}')

        return name

    def input_birthday(self, level=1, input_description='Input Birthday'):
        while True:
            birthday = self.input_(
                level,
                input_description + ' (format: YYYY-MM-DD): ')
            try:
                datetime.datetime.strptime(birthday, '%Y-%m-%d')
            except:
                pass
            else:
                print(self.tabs * (level+1), f'Your Input: {birthday!r}')
                return birthday

    def input_score(self, level=1, input_description='Input Score'):
        while True:
            score = self.input_(
                level,
                input_description + ' (format: integer): ')
            try:
                score = int(score)
                if 0 <= score <= 100:
                    print(self.tabs * (level+1), f'Your Input: {score!r}')
                    return score
            except:
                pass

    def input_options(self, opts, level=1, input_description='Input'):
        opts = list(map(lambda x: str(x), opts))
        opts = list(map(lambda x: x.upper(), opts))
        q_str = (
            input_description +
            ' (Choose among ' + '{!r}' + ', {!r}'*(len(opts) - 1) + '): '
        )

        while True:
            opt = self.input_(level, q_str.format(*opts))

            if opt.upper() in opts:
                print(self.tabs * (level+1), f'Your Input: {opt.upper()!r}')
                return opt

    def merge_list(self, new_list):
        self.student_list = self.student_list.append(new_list, sort=False, ignore_index=True)
        self.student_list.index = self.student_list.index + 1
        self.student_list.average = self.student_list[['midterm', 'finalterm']].mean(axis=1)
        self.student_list.grade = self.student_list.average.apply(self.calc_grade)

    def calc_grade(self, average):
        if 90 < average:
            return 'A'
        elif 80 < average <= 90:
            return 'B'
        elif 70 < average <= 80:
            return 'C'
        elif 60 < average <= 70:
            return 'D'
        else:
            return 'F'

    def find_student(self):
        print('How do you want to find the student?')
        opt = self.input_options(['id', 'name'])
        if opt.upper() == 'ID':
            id = self.input_id()
            return self.student_list[self.student_list.id == id]
        else:
            name = self.input_name()
            return self.student_list[self.student_list.name == name]

    def add_a_new_entry(self):
        id = self.input_id()
        name = self.input_name()
        birthday = self.input_birthday()
        midterm = self.input_score('Input Midterm Score')
        finalterm = self.input_score('Input Finalterm Score')

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
            print('Do you really want to delete?')
            opt = self.input_options(['y', 'n'])

            if opt.upper() == 'Y':
                self.student_list.drop(target_list.index, inplace=True)

                if len(self.student_list):
                    self.student_list.index = range(len(self.student_list))
                    self.student_list.index = self.student_list.index + 1

    def find_some_item_from_entry(self):
        target_list = self.find_student()

        print(target_list[['average', 'grade']])

    def modify_an_entry(self):
        target_list = self.find_student()

        print('Which test do you want to modify?')
        opt = self.input_options(['midterm', 'finalterm'])
        score = self.input_score()

        if opt.upper() == 'MIDTERM':
            self.student_list[self.student_list.index == target_list.index].midterm = score
        else:
            self.student_list[self.student_list.index == target_list.index].finalterm = score

    def print_the_contents_of_all_entries(self):
        if len(self.student_list):
            print(self.student_list)

    def read_personal_data(self):
        filename = input('Input Filename: ')
        try:
            new_list = pd.read_csv(
                filename,
                sep="\s+",
                names=['index'] + self.columns,
                index_col=['index'],
                parse_dates=['birthday'],
                dtype={'id':'object', 'grade':'object'}
            )

            self.merge_list(new_list)
        except pd.errors.EmptyDataError:
            print('The file is empty.', file=sys.stderr)
        except FileNotFoundError:
            print('The file doesn\'t exist.', file=sys.stderr)

    def sort_entries(self):
        if len(self.student_list):
            print(self.student_list.sort_values(by=['name', 'average', 'grade']))

    def write_the_contents_to_the_same_file(self):
        filename = input('Input Filename: ')

        with open(filename, 'w') as OUT:
            OUT.write(self.student_list.to_string(header=False, index_names=False))

    #def run(self):
    #    self.input_id()
    #    self.input_name()
    #    self.input_birthday()
    #    self.input_score()
    #    self.input_options(['a', 'b', 'c'])

if __name__ == '__main__':

    GradeManagementPandas().run()
