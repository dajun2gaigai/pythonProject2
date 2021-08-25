#分布式多进程：https://blog.csdn.net/u011318077/article/details/88094583
#廖：https://www.liaoxuefeng.com/wiki/1016959663602400/1017631559645600

'''
os.fork()
multiprocessing.Process()
class MyProcess(Process):
multiprocessing.set_start_method('spawn')
ctx = multiprocessing.get_context('spawn')
ctx.Process(target=_, args=(_,),name=_)
multiprocessing.Pool
subprocessing
multiprocessing.Queue multiprocessing.Pipe
multiprocessing.manager.BaseManager
'''

#调用其提供的 fork() 函数，程序中会拥有 2 个进程，其中父进程负责执行整个程序代码，
# 而通过 fork() 函数创建出的子进程，会从创建位置开始，执行后续所有的程序（包含创建子进程的代码）。
import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))

#multiprocessing.Process
from multiprocessing import Process
# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')

#创建进程类
from multiprocessing import Process
import os
print("当前进程ID：",os.getpid())
# 定义一个函数，供主进程调用
def action(name,*add):
    print(name)
    for arc in add:
        print("%s --当前进程%d" % (arc,os.getpid()))
#自定义一个进程类
class My_Process(Process):
    def __init__(self,name,*add):
        super().__init__()
        self.name = name
        self.add = add
    def run(self):
        print(self.name)
        for arc in self.add:
            print("%s --当前进程%d" % (arc,os.getpid()))
if __name__=='__main__':
    #定义为进程方法传入的参数
    my_tuple = ("http://c.biancheng.net/python/",\
                "http://c.biancheng.net/shell/",\
                "http://c.biancheng.net/java/")
    my_process = My_Process("my_process进程",*my_tuple)
    #启动子进程
    my_process.start()
    #主进程执行该函数
    action("主进程",*my_tuple)

#三种设置进程启动的方式 http://c.biancheng.net/view/2633.html
'''
spawn：使用此方式启动的进程，只会执行和 target 参数或者 run() 方法相关的代码。Windows 平台只能使用此方法，事实上该平台默认使用的也是该启动方式。相比其他两种方式，此方式启动进程的效率最低。
fork：使用此方式启动的进程，基本等同于主进程（即主进程拥有的资源，该子进程全都有）。因此，该子进程会从创建位置起，和主进程一样执行程序中的代码。注意，此启动方式仅适用于 UNIX 平台，os.fork() 创建的进程就是采用此方式启动的。
forserver：使用此方式，程序将会启动一个服务器进程。即当程序每次请求启动新进程时，父进程都会连接到该服务器进程，请求由服务器进程来创建新进程。通过这种方式启动的进程不需要从父进程继承资源。注意，此启动方式只在 UNIX 平台上有效。
'''
#两种手动设置进程的启动方法
#set_start_method() 该函数的调用位置，必须位于所有与多进程有关的代码之前。
import multiprocessing
import os

print("当前进程ID：", os.getpid())
# 定义一个函数，准备作为新进程的 target 参数
def action(name, *add):
    print(name)
    for arc in add:
        print("%s --当前进程%d" % (arc, os.getpid()))

if __name__ == '__main__':
    # 定义为进程方法传入的参数
    my_tuple = ("http://c.biancheng.net/python/", \
                "http://c.biancheng.net/shell/", \
                "http://c.biancheng.net/java/")
    # 设置进程启动方式
    #由于此程序中进程的启动方式为 spawn，因此该程序可以在任意（ Windows 和类 UNIX 上都可以 ）平台上执行。
    multiprocessing.set_start_method('spawn')

    # 创建子进程，执行 action() 函数
    my_process = multiprocessing.Process(target=action, args=("my_process进程", *my_tuple))
    # 启动子进程
    my_process.start()
#get_context(): 使用 multiprocessing 模块提供的 get_context() 函数来设置进程启动的方法，调用该函数时可传入 "spawn"、"fork"、"forkserver" 作为参数，用来指定进程启动的方式
#需要注意的一点是，前面在创建进程是，使用的 multiprocessing.Process() 这种形式，而在使用 get_context() 函数设置启动进程方式时，需用该函数的返回值，代替 multiprocessing 模块调用 Process()。
import multiprocessing
import os
print("当前进程ID：", os.getpid())
# 定义一个函数，准备作为新进程的 target 参数
def action(name, *add):
    print(name)
    for arc in add:
        print("%s --当前进程%d" % (arc, os.getpid()))
if __name__ == '__main__':
    # 定义为进程方法传入的参数
    my_tuple = ("http://c.biancheng.net/python/", \
                "http://c.biancheng.net/shell/", \
                "http://c.biancheng.net/java/")
    # 设置使用 fork 方式启动进程
    ctx = multiprocessing.get_context('spawn')

    # 用 ctx 代替 multiprocessing 模块创建子进程，执行 action() 函数
    my_process = ctx.Process(target=action, args=("my_process进程", *my_tuple))
    # 启动子进程
    my_process.start()

#multiprocessing.Pool
from multiprocessing import Pool
import os, time, random
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    #调用close表示进程池中不能添加其他子进程了。调用p.join()前要先调用p.close()
    p.close()
    p.join()
    print('All subprocesses done.')

#subprocessing
import subprocess
print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)

print('$ nslookup')
p = subprocess.Popen(['nslookup'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output, err = p.communicate(b'set q=mx\npython.org\nexit\n')
print(output.decode('utf-8'))
print('Exit code:', p.returncode)

#进程间通信Queue,Pipe
from multiprocessing import Process, Queue
import os, time, random
from multiprocessing import Process, Queue, Pipe
# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()



#分布式多进程 multiprocessing.manager.BaseManager
#task_server.py
import random, time, queue
from multiprocessing.managers import BaseManager

# 发送任务的队列:
task_queue = queue.Queue()
# 接收结果的队列:
result_queue = queue.Queue()

# 从BaseManager继承的QueueManager:
class QueueManager(BaseManager):
    pass

# 把两个Queue都注册到网络上, callable参数关联了Queue对象:
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
# 绑定端口5000, 设置验证码'abc':
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动Queue:
manager.start()
# 获得通过网络访问的Queue对象:
task = manager.get_task_queue()
result = manager.get_result_queue()
# 放几个任务进去:
for i in range(10):
    n = random.randint(0, 10000)
    print('Put task %d...' % n)
    task.put(n)
# 从result队列读取结果:
print('Try get results...')
for i in range(10):
    r = result.get(timeout=10)
    print('Result: %s' % r)
# 关闭:
manager.shutdown()
print('master exit.')

# task_worker.py
import time, sys, queue
from multiprocessing.managers import BaseManager

# 创建类似的QueueManager:
class QueueManager(BaseManager):
    pass

# 由于这个QueueManager只从网络上获取Queue，所以注册时只提供名字:
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

# 连接到服务器，也就是运行task_master.py的机器:
server_addr = '127.0.0.1'
print('Connect to server %s...' % server_addr)
# 端口和验证码注意保持与task_master.py设置的完全一致:
m = QueueManager(address=(server_addr, 5000), authkey=b'abc')
# 从网络连接:
m.connect()
# 获取Queue的对象:
task = m.get_task_queue()
result = m.get_result_queue()
# 从task队列取任务,并把结果写入result队列:
for i in range(10):
    try:
        n = task.get(timeout=1)
        print('run task %d * %d...' % (n, n))
        r = '%d * %d = %d' % (n, n, n*n)
        time.sleep(1)
        result.put(r)
    except Queue.Empty:
        print('task queue is empty.')
# 处理结束:
print('worker exit.')