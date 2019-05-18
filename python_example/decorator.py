class MyClass:
    def method(self):
        return 'instance method called'

    @classmethod
    def classmethod(cls):
        return 'class methond called', cls

    @staticmethod
    def staticmethod():
        return 'static method called'

obj = MyClass()
print(obj.method())
print(MyClass.method())
print(obj.classmethod())
print(obj.staticmethod())