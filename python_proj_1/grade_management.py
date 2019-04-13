import os
from student_credits_list import StudentCreditsList
from students import Students

class GradeManagement:
    """학점 관리 기본 클래스
    StudentCreditsList 클래스와 Students Import하여
    불러오기, 등록, 삭제의 기능을 수행한다.
    """
    def __init__(self):
        self._student_credits_list = StudentCreditsList()
        self._filename = ''
        pass

    def show_help_message(self):
        help_message = """
######################################################
(a) (‘A’ 또는 ‘a’) add a new entry
(d) (‘D’ 또는 ‘d’) delete an entry
(f) (‘F’ 또는 ‘f’) find some item from entry
(m) (‘M’ 또는 ‘m’) modify an entry
(p) (‘P’ 또는 ‘p’) print the contents of all entries
(r) (‘R’ 또는 ‘r’) read personal data from a file
(s) (‘S’ 또는 ‘s’) sort entries
(q) (‘Q’ 또는 ‘q’) quit
(w) (‘W’ 또는 ‘w’) write the contents to the same file
######################################################
        """
        print(help_message)



    def check_input(self):
        input_description = "Choose one of the options below(Help : h) :     "

        while True:
            try:
                input_string = input(input_description).upper()
                assert input_string in ['A', 'D', 'F', 'M', 'P', 'R', 'S', 'Q', 'W', 'H'], \
                    "You can only A, D, F, M, P, R, S, Q, W"
                return input_string
            except AssertionError as e:
                print(repr(e))

    def check_input_ext(self, input_description = '', prohibit_list = []):

        while True:
            try:
                input_string = input(input_description).upper()
                assert input_string in prohibit_list, \
                    "You can only {0}".format(str(prohibit_list))
                return input_string
            except AssertionError as e:
                print(repr(e))

    def check_input_filename(self):
        input_description = "현재 디렉토리에 있는 Data 파일을 입력하십시요. :    "
        while True:
            try:
                input_string = input(input_description)

                filenames = os.listdir(os.path.dirname(os.path.abspath(__file__)))
                data_file_list = [file for file in filenames if file == input_string]
                if len(data_file_list) ==1:
                    return data_file_list[0]
                    break
                else:
                    print("현재 디렉토리 파일이름 : {0}".format(str(filenames)))
                    raise FileNotFoundError("파일을 찾을수 없습니다.")
            except FileNotFoundError as e:
                print(repr(e))

    def add_a_new_entry(self):
        # id, 이름, 생년월일, 중간고사, 기말고사 ㅁ점수를 물어보도록 하고, user가 입력한 내용을 맨 밑줄에 새롭게 추가한다 (일련번호 추가 필요).

        # students에서 상속받는게 좋을듯

        student_data = Students()
        columns = student_data.columns()
        input_value = list()
        # Column에서 True인것만 가져옴
        input_column = [_k for _k, _v in columns.items() if _v is True]
        for column_val in input_column:
            while True:
                try:
                    input_value.append(input(column_val + " : "))
                    student_data.set_validation = input_value
                    break
                except ValueError as e:
                    input_value.pop()
                    print(e)
        self._student_credits_list.append(student_data)
        print(self._student_credits_list)

    def delete_an_entry(self):
        input_value = input("지울 대상의 ID나 이름을 입력하세요 :  ")

        # ID와 이름으로 지울 대상을 찾아서 해당 row의 index를 가져온다
        results = list(filter(lambda item: (item[1]._name == input_value or item[1]._id == input_value ),
                              enumerate(self._student_credits_list)))
        if len(results) == 0:
            print("지울 대상이 없습니다.")
        else:
            confirm_del = input("총 {0} 건이 검색되었습니다. 정말 지우시겠습니까? (Y/n)".format(str(len(results))))
            if confirm_del == 'Y':
                # List는 index가 큰거 부터 지운다
                for _r in sorted(results, reverse=True):
                    self._student_credits_list.remove(_r[0])
                    print("{0}({1}) 을(를) 지웠습니다.".format(_r[1]._name, _r[1]._id))
            else:
                print("지우기를 취소하였습니다.")
            # reordering
            for _idx, _item in enumerate(self._student_credits_list):
                _item.index = _idx

    def find_some_item_from_entry(self):
        #  F : ID와 이름으로 학생 찾기
        input_value = input("검색 할 대상의 ID나 이름을 입력하세요 :  ")
        # ID와 이름으로 대상을 찾아서 해당 row의 index를 가져온다
        results = list(filter(lambda item: (item[1]._name == input_value or item[1]._id == input_value),
                              enumerate(self._student_credits_list)))
        if len(results) == 0:
            print("검색 된 대상이 없습니다.")
        else:
            print("Total rows : {0}건".format(str(len(results))))
            for _r in results:
                print("{0}({1})님의 평균점수는 {2}, Grade는 {3} 입니다.".format(_r[1]._name,
                _r[1]._id, _r[1]._mean, _r[1]._grade ))

    def modify_an_entry(self):
        # F : ID와 이름으로 학생 찾기
        input_value = input("수정 할 대상의 ID나 이름을 입력하세요 :  ")
        # ID와 이름으로 대상을 찾아서 해당 row의 index를 가져온다
        results = list(filter(lambda item: (item[1]._name == input_value or item[1]._id == input_value),
                              enumerate(self._student_credits_list)))
        if len(results) == 0:
            print("검색 된 대상이 없습니다.")
        else:
            print("Total rows : {0}건".format(str(len(results))))
            print("###############################################")
            for _idx , _r in enumerate(results,1):
                print("[{0}/{1}] : {2}({3})님의 중간시험 점수 {4}, 기말시험 점수 {5} 는 입니다.".format(_idx, len(results),
                                                                                      _r[1]._name,
                _r[1]._id, _r[1]._midterm, _r[1]._finalterm ))
                prohibit_list = ['1', '2', 'N']
                confirm_modify = self.check_input_ext("[{0}/{1}] : {2}({3})님의 중간시험 점수를 수정하실려면 1 "
                                ", 기말시험 점수를 수정하실려면 2 수정을 안하시겠으면 n을 눌러주세요 (1/2/n)   :    "
                                .format(_idx, len(results),_r[1]._name,_r[1]._id), prohibit_list)

                if confirm_modify == '1':
                    while True:
                        mid_term_score = input("[{0}/{1}] : {2}({3})님의 중간고사 점수를 입력하세요 :   "
                                               .format(_idx, len(results),_r[1]._name,_r[1]._id,))
                        try:
                            _r[1].midterm = mid_term_score
                            break
                        except ValueError as e:
                            print(e)
                elif confirm_modify == '2':
                    while True:
                        final_term_score = input("[{0}/{1}] : {2}({3})님의 기말고사 점수를 입력하세요 :   "
                                               .format(_idx, len(results), _r[0], len(results),_r[1]._name,_r[1]._id,))
                        try:
                            _r[1].finalterm = final_term_score
                            break
                        except ValueError as e:
                            print(e)
                elif confirm_modify == 'n':
                    print("점수를 수정하지 않습니다.")
                print("[{0}/{1}] : 수정되었습니다. {2}".format(_idx, len(results), str(_r[1])))


    def print_the_contents_of_all_entries(self):
        if len(self._student_credits_list) == 0:
            print("데이터가 없습니다.")
        else:
            print(self._student_credits_list)

    def read_personal_data(self):
        self.file_name = self.check_input_filename()
        with open(self.file_name) as f:
            lines_all = f.readlines()
        try:
            data = [Students(line.replace('\n', '').split('\t')) for line in lines_all]
            self._student_credits_list = StudentCreditsList(data)
            print(self._student_credits_list)
        except Exception as e:
            print("Data file 을 읽다가 오류가 발생했습니다. [{0}]".format(e.__repr__()))

    def sort_entries(self):
        prohibit_list = ['N', 'A', 'G']
        confirm_modify = self.check_input_ext("정렬할 기준을 선택해주세요. 이름(n), 평균점수(a), Grade(g) (n/a/g) :  ",
                                              prohibit_list)
        if confirm_modify == 'N':
            self._student_credits_list.sort(key=lambda x: (x._name, x._mean), reverse=False)
        elif confirm_modify == 'A':
            self._student_credits_list.sort(key=lambda x: x._mean, reverse=True)
        elif confirm_modify == 'G':
            self._student_credits_list.sort(key=lambda x: (x._grade, x._mean), reverse=False)
        print(self._student_credits_list)

    def write_the_contents_to_the_same_file(self):
        self._student_credits_list.save("./"+self.file_name)
        print("{0} 에 저장되었습니다.".format(self.file_name))

    def run(self):
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
