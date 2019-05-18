"""
Number 2 answer

"""

import pandas as pd

df = pd.read_csv("phonebook.txt")

def number_search(name):
    """
    Find number using value of name
    """
    print(df[df['name']==name])

def region_book(area_number):
    """
    Find store info using value of area number.
    """
    print(df[df['tel'].str.match(area_number+"-")])

if __name__ == "__main__":
    """
    Input name and find it
    Input area code and find it using pandas
    """

    name = input("Enter Name : ")
    number_search(name)
    area = input("Enter area code : ")
    region_book(area)