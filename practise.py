import os
import sys

from pathlib import Path
from pathlib import PurePosixPath

#相当于Path(__file__)
p = Path('.').absolute()
#拼接路径
q = p / 'ab'
print(q)
print(Path(p,'ab'))

# print(q.exists())
# print(q.is_dir())
# with q.open() as f: f.readline()
#
# x = [i for i in p.iterdir() if i.is_dir()]
# print(x)
# print(list(p.glob('**/*.py')))
p = PurePosixPath('/home/u910')
print(p.root)
print(p.parent)
print([x for x in p.parents])

print(sys.path)
