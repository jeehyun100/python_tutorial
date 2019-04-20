import argparse
from grade_management import GradeManagement
from grade_management_pandas import GradeManagementPandas

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
