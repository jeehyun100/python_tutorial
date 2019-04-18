import os
from student_credits_list import StudentCreditsList
from students import Students

class GradeManagement:
    """학점 관리 기본 클래스
    # StudentCreitsList 클래스와 Students Import하여
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
        pass

    def delete_an_entry(self):
        pass

    def find_some_item_from_entry(self):
        #  F : ID와 이름으로 학생 찾기
        pass

    def modify_an_entry(self):
        input_description_1="(수정모드) [ID] 또는 [이름]을 하시오 : "
        input_description_2="(수정모드) [중간점수](1) 또는 [기말점수](2)를 선택하시오 : "
        while True:
            try:
                input_string_1 = input(input_description_1)
                with open("data.txt",'r+',encoding='UTF-8') as f:
                    lines_all = f.readlines()
                    student = [Students(line.replace('\n','').split('\t')) for line in lines_all]
                    student_obj=StudentCreditsList(student)
                    for student_list in student_obj:
                        if (input_string_1 == student_list._id) or (input_string_1 == student_list._name) :
                            print("\n[학생 ID: {0}  이름: {1}  중간고사: {2} 기말고사: {3}]\n".format(student_list._id,student_list._name,student_list._midterm,student_list._finalterm))
                            input_string_2 = int(input(input_description_2))
                            if(input_string_2 ==1):
                                input_string_mid = input('\n(수정) 중간 점수를 입력하십시오 : ')
                                student_list._midterm = input_string_mid
                                print(student_list)
                                break
                            elif(input_string_2 ==2):
                                input_string_final = input('\n(기말) 중간 점수를 입력하십시오 : ')
                                student_list._finalterm = input_string_final
                                print(student_list)
                                break

                print(student_obj)
                StudentCreditsList.save(student_obj,"./data.txt")

            except FileNotFoundError as e:
                print(repr(e))
            #self.print_the_contents_of_all_entries();


            # First Load datafile
            # r
            #
            # ID와 이름으로 대상을 찾아서 해당 row의 index를 가져온다
            # results = list(filter(lambda item: (item[1]._name == input_value or item[1]._id == input_value),enumerate(self._student_credits_list)))
            #
            # how to access student class
            # for _idx, _r in enumerate(results, 1):
            #     print("[{0}/{1}] : {2}({3})님의 중간시험 점수 {4}, 기말시험 점수 {5} 는 입니다.".format(_idx, len(results),
            #                                                                          _r[1]._name,
            # how to save
            # _student_credits_list.save()


    def print_the_contents_of_all_entries(self):
        with open("data.txt",encoding='UTF-8') as f:
            lines_all = f.readlines()
        print_s = [ Students(line.replace('\n','').split('\t')) for line in lines_all] #studnet 객체 생성
        print_str_row = StudentCreditsList(print_s)
        print(print_str_row)

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
        with open("data.txt",encoding='UTF-8') as f:
            lines_all = f.readlines()
        print_s = [ Students(line.replace('\n','').split('\t')) for line in lines_all] #studnet 객체 생성
        print_str_row = StudentCreditsList(print_s)
        list=[]
        name_list=[]
        name_sort=[]
        for student_list in print_str_row:
            list.append(student_list)
        print(list[0]._name)
        input_description_1="(정렬모드) 이름순?(n), 평균점수순?(a), grade순?(g) : "
        for i in range(19):
            name_list.append(list[i]._name)
        sort=sorted(name_list)
        for x in range(19):
            for y in range(19):
                if(str(sort[x])==list[y]._name):
                    name_sort.append(list[y])
        print(name_sort)



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
