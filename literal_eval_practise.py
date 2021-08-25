#https://blog.csdn.net/sinat_33924041/article/details/88350569
from ast import literal_eval

eval('2+5*4') #22
d="{'name':'python','age':20}"
dd = eval(d)
type(dd)

l = "[2,3,4,5]"
ll = eval(l)

t='(1,2,3)'
tt = eval(t)

#ast.literal_eval:与eval()一样，只是会解析是否符合python语法，更安全
res = literal_eval('1 + 1')
print(res)
