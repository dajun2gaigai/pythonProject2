#https://zhuanlan.zhihu.com/p/37239021
#这些类型注解加不加，对你的代码来说没有任何影响：
#让别的程序员看懂，让IDE了解变量类型，提供更好的代码提示，不全和语法检查

def add(x:int, y:int) -> int:
    return x+y

print(add(1,4))
print(add('hell','worl'))
print(add(1.3,1.4))
