.. python_proj_1 documentation master file, created by
   sphinx-quickstart on Thu Apr 18 13:23:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

자료 관리 프로그램 문서화
======================================================

**백지현, 지동근, 김후빈**

필수사항 ::

   pandas >= 0.24.2

실행방법 ::

    Standard version :
        python run.py
    Pandas version:
        python run.py -p

작업내용 ::

    설계
        Standard version : 백지현
        Pandas version : 지동근
    구현
        Standard vesion : 백지현, 김후빈
        Pandas version : 지동근

    테스트 : 김후빈
    문서화 : 백지현, 지동근, 김후빈


프로그램 설명 ::

   Program을 시작하면 아래의 메시지를 screen에 display하고 사용자의 명령어(아래에 주어진 9가지)를
   따라서 해당 업무를 수행하도록 한다.

   ===================================
   Choose one of the options below : _
   ===================================

   학점평가 관리 프로그램은 다음의 10가지 기능
   (a) (‘A’ 또는 ‘a’) add a new entry
   -> id, 이름, 생년월일, 중간고사, 기말고사 점수를 물어보도록 하고, user가 입력한 내용을 맨 밑줄에
   새롭게 추가한다 (일련번호 추가 필요).

   (d) (‘D’ 또는 ‘d’) delete an entry
   -> id 혹은 이름을 물어본 후, user가 입력한 내용의 entry를 삭제하고, 일련번호를 수정한다.
   (단, 삭제를 원하는지 한 번 더 확인하는 절차를 거친다.)

   (f) (‘F’ 또는 ‘f’) find some item from entry
   -> id 나 이름을 물어보고, 입력한 학생의 평균점수와 grade를 알려준다.

   (m) (‘M’ 또는 ‘m’) modify an entry
   -> id나 이름을 물어보고, 입력한 학생의 중간시험 또는 기말시험 점수 중 어느 것을 수정할지 물어보고,
   해당 학생의 점수를 새로이 입력한 후, 확인용으로 프린트한다.

   (p) (‘P’ 또는 ‘p’) print the contents of all entries
   -> 일렬번호부터 grade 까지 모든 정보를 순서대로 프린트한다.
   단, 보기 좋게 하기 위해 한 학생 정보는 한 줄에 표시한다.

   (r) (‘R’ 또는 ‘r’) read personal data from a file
   -> 파일 이름을 입력한 후, 해당 파일에서 개인정보(id, 이름, 생년월일)를 읽어들인다.

   (s) (‘S’ 또는 ‘s’) sort entries
   -> 이름순?(n), 평균점수순?(a), grade순?(g) (이때 평균점수가 높은 순서도 고려)를 물어보고,
   해당 칸의 정보를 순서대로(내림차순) 정렬하고, 이를 print한다.
   a을 입력받으면, 높은 순서대로 모든 정보(일련번호, 학생 id, ......)를 출력한다.

   (q) (‘Q’ 또는 ‘q’) quit
   -> 프로그램 동작을 마친다.

   (w) (‘W’ 또는 ‘w’) write the contents to the contents to the same file
   -> 현재 내용을 해당 파일에 저장한다.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   python_proj_1


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
