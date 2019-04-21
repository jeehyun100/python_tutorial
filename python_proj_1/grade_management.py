import os
import datetime
from student_credits_list import StudentCreditsList
from students import Students
from students0 import Students0


class GradeManagement:
    """ Manage the student's credits

    The "GradeManagement" class manage student's credits.
    This class have functions add, delete, find, modify, print
    , read, sort, quit, write files.

    """
    def __init__(self):
        """Creates a "GradeManagement"
           It creates "StudentCreditsList()" and filename variable for saving.

        """
        self._student_credits_list = StudentCreditsList()
        self._filename = None
        self.tabs = '  '

    def show_help_message(self):
        """Show help message

        """

        help_message = """
######################################################
(a) (‘A’ or ‘a’) add a new entry
(d) (‘D’ or ‘d’) delete an entry
(f) (‘F’ or ‘f’) find some item from entry
(m) (‘M’ or ‘m’) modify an entry
(p) (‘P’ or ‘p’) print the contents of all entries
(r) (‘R’ or ‘r’) read personal data from a file
(s) (‘S’ or ‘s’) sort entries
(q) (‘Q’ or ‘q’) quit
(w) (‘W’ or ‘w’) write the contents to the same file
######################################################
        """
        print(help_message)

    def input_(self, level=0, input_description=''):
        tabs = self.tabs * level
        return input(tabs + input_description)

    def input_id(self, level=1, input_description='Input ID'):
        while True:
            id = self.input_(
                level,
                input_description + ' (format: XXXXXXXX): ')

            if len(id) == 8:
                print(self.tabs * (level+1), f'Your Input: {id!r}')
                return id
            else:
                print(self.tabs * (level+1), "Wrong input.")

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
                print(self.tabs * (level+1), "Wrong input.")
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
                print(self.tabs * (level+1), "Wrong input.")

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
            else:
                print(self.tabs * (level+1), "Wrong input.")

    def input_filename(self, level=1, input_description='Input Filename'):
        """The function to check if the file exists

        Return:
             A type `string` Name if first match file
        Raises:
            FileNotFoundError: If the file is not existing in the directory.
        """
        while True:
            filename = self.input_(
                level,
                input_description + ' : ')

            if os.path.isfile(filename):
                print(self.tabs * (level+1), f'Your Input: {filename!r}')
                return filename
            else:
                print(self.tabs * (level+1), "The file doesn't exist.")

    def check_input(self):
        """Check input value
        You can only input A, D, F, M, P, R, S, Q, W.

        Raises:
            ValueError: If input value is other than indicated above.

        """
        input_description = "Choose one of the options below (Help : h) : "

        while True:
            try:
                input_string = input(input_description).upper()
                assert input_string in ['A', 'D', 'F', 'M', 'P', 'R', 'S', 'Q', 'W', 'H'], \
                    "You can only input A, D, F, M, P, R, S, Q, W, H"
                return input_string
            except AssertionError as e:
                print(repr(e))

    def check_input_ext(self, input_description='', prohibit_list=[]):
        """Check input value using prohibit list

        Args:
            input_description: The description text for input value.
            prohibit_list: Ths list of prohibit charactors

        For example:
            check_input_ext('Please Input only Y, ['Y'])

        Raises:
            ValueError: If input value has the wrong.

        """

        while True:
            try:
                input_string = input(input_description).upper()
                assert input_string in prohibit_list, \
                    "You can only {0}".format(str(prohibit_list))
                return input_string
            except AssertionError as e:
                print(repr(e))

    def attach_index(self):

        for i, student in enumerate(self._student_credits_list, start=1):
            student.index = i

    def add_a_new_entry(self):
        """The function to check if a file is in the directory

        Return:
             A type `string` Name if first match file
        Raises:
            FileNotFoundError: If file is not existing in the directory.

        """
        id = self.input_id()
        name = self.input_name()
        birthday = self.input_birthday()
        midterm = self.input_score(1, 'Input Midterm Score')
        finalterm = self.input_score(1, 'Input Finalterm Score')

        self._student_credits_list.append(
            Students((0, id, name, birthday, midterm, finalterm)))

        self.attach_index()

    def find_student(self):

        opt = self.input_options(['id', 'name'], 1, 'How do you want to find the student?')
        if opt.upper() == 'ID':
            id = self.input_id(1, "Input ID of Student You're Looking for")
            return (
                filter(lambda item: item._id == id, self._student_credits_list)
            )
        else:
            name = self.input_name(1, "Input Name of Student You're Looking for")
            return (
                filter(lambda item: item._name == name, self._student_credits_list)
            )

    def delete_an_entry(self):

        for student in self.find_student():
            self._student_credits_list.remove(student)

    def find_some_item_from_entry(self):

        students = self.find_student()
        print('{:10s}{:10s}{}'.format('일련번호', '평균', 'Grade'))
        for student in students:
            print('{:<10d}{:<10.2f}{:<10s}'.format(student.index, student.mean, student.grade))

    def modify_an_entry(self):
        """The function to modify midterm or finalterm data of students in list

        For example:
            1) to modify data, input student ID or name data
            2) choice a midterm or finalterm for the student
            3) Enter the student's score
        """

        students = self.find_student()

        opt = self.input_options(['midterm', 'finalterm'], 1, 'Which test do you want to modify?')
        score = self.input_score()

        if opt.upper() == 'MIDTERM':
            for student in students:
                student.midterm = score
        else:
            for student in students:
                student.finalterm = score

    def print_the_contents_of_all_entries(self):
        """The function to print all student credits in the memory

        """
        if len(self._student_credits_list):
            print(self._student_credits_list)
        else:
            print('There is no contents to show')

    def read_personal_data(self):
        """The function to read data file which is student credit data.

            Recreate "StudentCreditsList" from the file.

        Raises:
            Exception: While reading file has wrong.

        """

        self._filename = self.input_filename()
        with open(self._filename, encoding='utf-8') as f:
            lines_all = f.readlines()
        try:
            for line in lines_all:
                self._student_credits_list.append(Students(line.split()))
        except Exception as e:
            print("Data file 을 읽다가 오류가 발생했습니다. [{0}]".format(e.__repr__()))

        self.attach_index()
        self.print_the_contents_of_all_entries()

    def sort_entries(self):
        """The function to print sorted all student credits in the memory
        the sorting method is name, mean, grade.(

        """
        if not len(self._student_credits_list):
            print('There is no contents to sort')
            return

        # Do not change the order of list in place
        student_list_for_print = StudentCreditsList()
        for student in self._student_credits_list:
            student_list_for_print.append(student)

        opt = self.input_options(['n', 'a', 'g'], 1, 'Sort by name(n) or average(a) or grade(g)')
        if opt.upper() == 'N':
            student_list_for_print.sort(key=lambda x: x.mean, reverse=True)
            student_list_for_print.sort(key=lambda x: x.name)
        elif opt.upper() == 'A' or opt.upper() == 'G':
            student_list_for_print.sort(key=lambda x: x.name)
            student_list_for_print.sort(key=lambda x: x.mean, reverse=True)

        print(student_list_for_print)

    def write_the_contents_to_the_same_file(self):
        """The function to save all student credits

        """
        if not len(self._student_credits_list):
            print('There is no contents to write')
            return

        if self._filename is None:
            self._filename = self.input_filename()

        self._student_credits_list.save(self._filename)
        print(f'Data are saved into {self._filename!r}')

    def run(self):
        """The function to run to start the program.

        """
        while True:
            input_string = self.check_input()
            if input_string.upper() == 'A':
                self.add_a_new_entry()
            elif input_string.upper() == 'D':
                self.delete_an_entry()
            elif input_string.upper() == 'F':
                self.find_some_item_from_entry()
            elif input_string.upper() == 'M':
                self.modify_an_entry()
            elif input_string.upper() == 'P':
                self.print_the_contents_of_all_entries()
            elif input_string.upper() == 'R':
                self.read_personal_data()
            elif input_string.upper() == 'S':
                self.sort_entries()
            elif input_string.upper() == 'Q':
                # Quit the program
                break
            elif input_string.upper() == 'W':
                self.write_the_contents_to_the_same_file()
            elif input_string.upper() == 'H':
                self.show_help_message()


if __name__ == '__main__':
    GradeManagement().run()
