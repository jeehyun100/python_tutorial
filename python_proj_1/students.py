import datetime

class Students:
    """ 학생 클래스
    학생에 관한 여러 프로퍼티 일련번호, 학생id, 이름, 생년월일, 중간고사, 기말고사, 평균, Grade를 가지고 있고,
    각각의 Property의 입력값을 validation 해준다
    """

    def __init__(self, *args):
        if len(args) != 0:
            data = args[0]
            self._index = data[0]
            self._id = data[1]
            self._name = data[2]
            self._birthday = data[3]
            self._midterm = int(data[4])
            self._finalterm = int(data[5])
            self._mean = (self._midterm + self._finalterm)/2
            self._grade = 'S' if self._mean>=95 \
                    else 'B' if (95 > self._mean and 90 <= self._mean) \
                    else 'C' if (90 > self._mean and 80 <= self._mean) \
                    else 'D' if (80 > self._mean and 70 <= self._mean)  \
                    else 'F'
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
        return "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
            str(self._index).ljust(8, " "), str(self._id).ljust(8, " "),
            str(self._name).ljust(8, " "), str(self._birthday).ljust(8, " "),
            str(self._midterm).ljust(8, " "), str(self._finalterm).ljust(8, " "),
            str(self._mean).ljust(8, " "), str(self._grade).ljust(8, " "))

    __repr__ = __str__

    def cal_mean_grade(self):
        self._mean = (self._midterm + self._finalterm) / 2
        self._grade = 'S' if self._mean >= 95 \
            else 'B' if (95 > self._mean and 90 <= self._mean) \
            else 'C' if (90 > self._mean and 80 <= self._mean) \
            else 'D' if (80 > self._mean and 70 <= self._mean) \
            else 'F'


    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def midterm(self):
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
        return self._finalterm

    @finalterm.setter
    def finalterm(self, value):
        try:
            self._finalterm = int(value)
            self.cal_mean_grade()
        except ValueError as e:
            raise ValueError(str(e)+" for {0} column ".format('finalterm'))

    @property
    def valid_set(self):
        return self._id

    @valid_set.setter
    def set_validation(self, values):
        try:
            for _i, _v in enumerate(values):
                if _i == 0:  # ID validation
                    self._id = str(_v)
                if _i == 1:  # Name Validation
                    self._name = str(_v)
                if _i == 2:  # BirthDay Validation
                    self._birthday = datetime.datetime.strptime(_v, '%Y-%m-%d').strftime('%Y-%m-%d')
                if _i == 3:  # Midterm Validation
                    self._midterm = int(_v)
                if _i == 4:  # Final Validation
                    self._finalterm = int(_v)
                    self.cal_mean_grade()
        except ValueError as e:
            raise ValueError(str(e)+" for {0} column ".format(str(self.columns()[_i])))

    @classmethod
    def columns(cls):
        # 입력 받아야 할 컬럼과 보여주는 컬럼을 설정함
        return {"일련번호": False, "학생 id": True, "이름": True, "생년월일": True,
                "중간고사": True, "기말고사": True, "평균": False, "Grade": False}
