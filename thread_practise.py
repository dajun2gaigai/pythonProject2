#http://c.biancheng.net/view/2629.html

'''
thread = threading.Thread(target=_, args=_, name=_); thread.start(), thread.join()
lock = threading.Lock(); lock.acquire(), lock.release()
local = threading.Local(); local.resource = res
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=4)
future = pool.submit(target=_, args=_)
future.done(); future.result(); future.add_done_callback(func, args=_)
pool.shutdown()
threading.Timer(10.0, func)

from sched import scheduler
'''

import time, threading

# 新线程执行的代码:
def loop():
    print('thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)

print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop, name='LoopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)

#多线程锁
import time, threading

# 假定这是你的银行存款:
balance = 0

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(2000000):
        change_it(n)

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)

#加锁
balance = 0
lock = threading.Lock()

def run_thread(n):
    for i in range(100000):
        # 先要获取锁:
        lock.acquire()
        try:
            # 放心地改吧:
            change_it(n)
        finally:
            # 改完了一定要释放锁:
            lock.release()

#创建自己的线程类
#在创建 Thread 类的子类时，必须重写从父类继承得到的 run() 方法。因为该方法即为要创建的子线程执行的方法，其功能如同第一种创建方法中的 action() 自定义函数。
import threading
class my_thread(threading):
    def __init__(self, add):
        threading.Thread.__init__(self)
        self.add = add
    def run(self):
        for arc in self.add:
            pass

import threading
#创建子线程类，继承自 Thread 类
class my_Thread(threading.Thread):
    def __init__(self,add):
        threading.Thread.__init__(self)
        self.add = add
    # 重写run()方法
    def run(self):
         for arc in self.add:
            #调用 getName() 方法获取当前执行该程序的线程名
            print(threading.current_thread().getName() +" "+ arc)
#定义为 run() 方法传入的参数
my_tuple = ("http://c.biancheng.net/python/",\
            "http://c.biancheng.net/shell/",\
            "http://c.biancheng.net/java/")
#创建子线程
mythread = my_Thread(my_tuple)
#启动子线程
mythread.start()
#主线程执行此循环
for i in range(5):
    print(threading.current_thread().getName())

#守护线程
#类线程的特点是，当程序中主线程及所有非守护线程执行结束时，未执行完毕的守护线程也会随之消亡
#程序中，子线程thread为daemon线程，当主线程执行完毕后，thread也就不继续执行了
import threading
import time
#定义线程要调用的方法，*add可接收多个以非关键字方式传入的参数
def action(length):
    for arc in range(length):
        time.sleep(1.0)
        #调用 getName() 方法获取当前执行该程序的线程名
        print(threading.current_thread().getName()+" "+str(arc))
#创建线程
thread = threading.Thread(target = action,args =(20,))
#将thread设置为守护线程
thread.daemon = True
#启动线程
thread.start()
#主线程执行如下语句
for i in range(5):
    print(threading.current_thread().getName())

#线程池
from concurrent.futures import ThreadPoolExecutor
import threading
import time
# 定义一个准备作为线程任务的函数
def action(max):
    my_sum = 0
    for i in range(max):
        print(threading.current_thread().name + '  ' + str(i))
        my_sum += i
    return my_sum
# 创建一个包含2条线程的线程池
pool = ThreadPoolExecutor(max_workers=2)
# 向线程池提交一个task, 50会作为action()函数的参数
future1 = pool.submit(action, 10)
# 向线程池再提交一个task, 100会作为action()函数的参数
future2 = pool.submit(action, 20)
# 判断future1代表的任务是否结束
print(future1.done())
time.sleep(3)
# 判断future2代表的任务是否结束
print(future2.done())
# 查看future1代表的任务返回的结果
print(future1.result())
# 查看future2代表的任务返回的结果
print(future2.result())
# 关闭线程池
pool.shutdown()

#前面程序调用了 Future 的 result() 方法来获取线程任务的运回值，但该方法会阻塞当前主线程，只有等到钱程任务完成后，result() 方法的阻塞才会被解除。
#如果程序不希望直接调用 result() 方法阻塞线程，则可通过 Future 的 add_done_callback() 方法来添加回调函数，该回调函数形如 fn(future)。
# 当线程任务完成后，程序会自动触发该回调函数，并将对应的 Future 对象作为参数传给该回调函数
#程序可以使用 with 语句来管理线程池，这样即可避免手动关闭线程池
from concurrent.futures import ThreadPoolExecutor
import threading
import time
# 定义一个准备作为线程任务的函数
def action(max):
    my_sum = 0
    for i in range(max):
        # print(threading.current_thread().name + '  ' + str(i))
        my_sum += i
    return my_sum
# 创建一个包含2条线程的线程池
with ThreadPoolExecutor(max_workers=2) as pool:
    # 向线程池提交一个task, 50会作为action()函数的参数
    future1 = pool.submit(action, 50)
    # 向线程池再提交一个task, 100会作为action()函数的参数
    future2 = pool.submit(action, 100)
    def get_result(future):
        print(future.result())
    # 为future1添加线程完成的回调函数
    future1.add_done_callback(get_result)
    # 为future2添加线程完成的回调函数
    future2.add_done_callback(get_result)
    print('--------------')

#ThreadPoolExecutor.map(func. *iterable)
#使用 map() 方法来启动 3 个线程
from concurrent.futures import ThreadPoolExecutor
import threading
import time
# 定义一个准备作为线程任务的函数
def action(max):
    my_sum = 0
    for i in range(max):
        # print(threading.current_thread().name + '  ' + str(i))
        my_sum += i
    return my_sum
# 创建一个包含4条线程的线程池
with ThreadPoolExecutor(max_workers=4) as pool:
    # 使用线程执行map计算
    # 后面元组有3个元素，因此程序启动3条线程来执行action函数
    results = pool.map(action, (50, 100, 150))
    print('--------------')
    for r in results:
        print(r)

#threading.local()
#如果多线程之间需要共享资源（如多人操作同一银行账户的例子），就使用互斥锁同步机制；反之，如果仅是为了解决多线程之间的共享资源冲突，则推荐使用线程局部变量。
import threading
# 创建全局ThreadLocal对象:
local = threading.local()
def process():
    # 3.获取当前线程关联的resource:
    res = local.resource
    print (res + "http://c.biancheng.net/python/")
def process_thread(res):
    # 1.为当前线程绑定ThreadLocal的resource:
    local.resource = res
    # 2.线程的调用函数不需要传递res了，可以通过全局变量local访问
    process()
t1 = threading.Thread(target=process_thread, args=('t1线程',))
t2 = threading.Thread(target=process_thread, args=('t2线程',))
t1.start()
t2.start()

#threading.Timer
#Thread 类有一个 Timer子类，该子类可用于控制指定函数在特定时间内执行一次。
from threading import Timer
def hello():
    print("hello, world")
# 指定10秒后执行hello函数
t = Timer(10.0, hello)
t.start()
#如果程序想取消 Timer 的调度，则可调用 Timer 对象的 cancel() 函数
from threading import Timer
import time
# 定义总共输出几次的计数器
count = 0
def print_time():
    print("当前时间：%s" % time.ctime())
    global t, count
    count += 1
    # 如果count小于10，开始下一次调度
    if count < 10:
        t = Timer(1, print_time)
        t.start()
# 指定1秒后执行print_time函数
t = Timer(1, print_time)
t.start()

#多线程任务调度scheduler
import threading
from sched import scheduler
def action(arg):
    print(arg)
#定义线程要调用的方法，*add可接收多个以非关键字方式传入的参数
def thread_action(*add):
    #创建任务调度对象
    sche = scheduler()
    #定义优先级
    i = 3
    for arc in add:
        # 指定1秒后执行action函数, scheduler.enter(delay, priority, action, argument, kwargs)
        #在 delay规定的时间后，执行 action 参数指定的函数，其中 argument 和 kwargs 负责为 action 指定的函数传参，
        # priority 参数执行要执行任务的等级，当同一时间点有多个任务需要执行时，等级越高（ priority 值越小）的任务会优先执行。该函数会返回一个 event，可用来取消该任务。
        sche.enter(1, i, action,argument=(arc,))
        i = i - 1
    #执行所有调度的任务
    #运行所有需要调度的任务。如果调用该方法的 blocking 参数为 True，该方法将会阻塞线程，直到所有被调度的任务都执行完成。
    sche.run()
#定义为线程方法传入的参数
my_tuple = ("http://c.biancheng.net/python/",\
            "http://c.biancheng.net/shell/",\
            "http://c.biancheng.net/java/")
#创建线程
thread = threading.Thread(target = thread_action,args =my_tuple)
#启动线程
thread.start()