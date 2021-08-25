#https://blog.csdn.net/wjqhit/article/details/103095729
import numpy as np
import datetime

print('%o' % 20)
print('%d' % 20)
print('%x' % 20)

print('%5.3f' % 1.2222) #前面表示站位几个字符，后面表示小数点保留几位
print('%.2e' % 2,33333)
print('%g' % 33322233.444444444) #正常小于6位整数，否则用科学计数法
print('%.2g' % 21324.4444)

round(1.1125)
round(1.1125)  # 四舍五入，不指定位数，取整 1
print(round(1.1135,3))  # 取3位小数，由于3为奇数，则向下“舍”, 1.113
print(round(1.1145,3)) #取3位小数，由于4为偶数，则向上“入”

#字符串
print('%s' % 'hello world')  # 字符串输出
print('%20s' % 'hello world')  # 右对齐，取20位，不够则补位
print('%-20s' % 'hello world')  # 左对齐，取20位，不够则补位
print('%.2s' % 'hello world')  # 取2位
print('%10.2s' % 'hello world')  # 右对齐，取2位
print('%-10.2s' % 'hello world')  # 左对齐，取2位

#format

print('{} {}'.format('hello','world'))  # 不带字段
print('{0} {1}'.format('hello','world'))  # 带数字编号
print('{0} {1} {0}'.format('hello','world'))  # 打乱顺序
print('{1} {1} {0}'.format('hello','world'))
print('{a} {tom} {a}'.format(tom='hello',a='world'))  # 带关键字

coord = (3, 5)
print('X: {0[0]};  Y: {0[1]}'.format(coord))

a = {'a': 'test_a', 'b': 'test_b'}
print('a:{0[a]}, b:{0[b]}'.format(a))
print("int: {0:d};  hex: {0:x};  oct: {0:o};  bin: {0:b}".format(42))
print("int: {0:d};  hex: {0:#x};  oct: {0:#o};  bin: {0:#b}".format(42)) #在前面加“#”，则带进制前缀

print('{} and {}'.format('hello', 'world'))  # 默认左对齐hello and world
print('{:10s} and {:>10s}'.format('hello', 'world'))  # <为默认，左对齐， >为右对齐 取10位左对齐，取10位右对齐hello and world
print('{:^10s} and {:^10s}'.format('hello', 'world'))  # 取10位中间对齐hello and world
print('{} is {:.2f}'.format(1.123, 1.123))  # 取2位小数1.123 is 1.12
print('{0} is {0:>10.2f}'.format(1.123))  # 取2位小数，右对齐，取10位1.123 is 1.12

'{:<30}'.format('left aligned')  # 左对齐
'left aligned                  '
'{:>30}'.format('right aligned')  # 右对齐
'                 right aligned'
'{:^30}'.format('centered')  # 中间对齐
'           centered           '
'{:*^30}'.format('centered')  # 使用“*”填充
'***********centered***********'
'{:0=30}'.format(11)  # 还有“=”只能应用于数字，这种方法可用“>”代替,右对齐


'{:+f}; {:+f}'.format(3.14, -3.14)  # 总是显示符号
'+3.140000; -3.140000'
'{: f}; {: f}'.format(3.14, -3.14)  # 若是+数，则在前面留空格
' 3.140000; -3.140000'
'{:-f}; {:-f}'.format(3.14, -3.14)  # -数时显示-，与'{:f}; {:f}'一致
'3.140000; -3.140000'


d = datetime.datetime(2010, 7, 4, 12, 15, 58)
'{:%Y-%m-%d %H:%M:%S}'.format(d)
'2010-07-04 12:15:58'

'{:,}'.format(1234567890)
'1,234,567,890'

#format变形

a = "hello"
b = "world"
f"{a} {b}"
'hello world'

name = 'jack'
age = 18
sex = 'man'
job = "IT"
salary = 9999.99

print(f'my name is {name.capitalize()}.')
print(f'I am {age:*^10} years old.')
print(f'I am a {sex}')
print(f'My salary is {salary:10.3f}')
print(f'my name is {name}')

# 结果
'''
my name is Jack.
I am ** ** 18 ** ** years old.
I am a man
My salary is 9999.990'''



