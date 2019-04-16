import os
import sys
import datetime
import argparse
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

    def input_id(self):
        while True:
            id = input('Input ID (format: XXXXXXXX): ')
            
            if len(id) == 6:
                return id

    def input_name(self):
        name = input('Input Name: ')

        return name

    def input_birthday(self):
        while True:
            birthday = input('Input Birthday (format: YYYY-MM-DD): ')
            try:
                datetime.datetime.strptime(birthday, '%Y-%m-%d')
            except:
                pass
            else:
                return birthday

    def input_score(self, test_type):
        while True:
            score = input(f'Input {test_type} Score (format: integer): ')
            try:
                score = int(score)
                if 0 <= score <= 100:
                    return score
            except:
                pass

    def input_options(self, opts):
        opts = list(map(lambda x: str(x), opts))
        opts = list(map(lambda x: x.upper(), opts))
        q_str = 'Input (Choose among ' + '{!r}' + ', {!r}'*(len(opts) - 1) + '): '
        
        while True:
            opt = input(q_str.format(*opts))

            if opt.upper() in opts:
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
        midterm = self.input_score('Midterm')
        finalterm = self.input_score('Finalterm')

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
        score = self.input_score('')

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

    def run(self):
        self.read_personal_data()
        self.sort_entries()
        self.print_the_contents_of_all_entries()

def main(args):
    if args.pandas:
        GradeManagementPandas().run()
    else:
        GradeManagement().run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pandas", action="store_true",
            help="output label images")
    args = parser.parse_args()

    main(args)
