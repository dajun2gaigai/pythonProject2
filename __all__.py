#只有以“from 模块名 import *”形式导入的模块，当该模块设有 __all__ 变量时，只能导入该变量指定的成员，未指定的成员是无法导入的。
#http://c.biancheng.net/view/2401.html
from __all__demo import *
import __all__demo as alld

say()
disPython()

alld.__CLanguage()