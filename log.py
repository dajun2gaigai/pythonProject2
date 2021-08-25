#https://www.cnblogs.com/pycode/p/logging.html
import logging

import logging
'''level: log输出级别，level低于这个level不输出：debug, info, warning, error, critic
通过下面的方式进行简单配置输出方式与日志级别, level配置默认level
filemode更改log写入方式：默认‘a’:append, 'w'：覆盖
format指定log输出出了msg意外一些其他内容，如%(asctime)s: %(levelname)s:%(filename)s:%(message)s输出三个level,file,msg内容'''
logging.basicConfig(filename='logger.log', filemode='w', level=logging.DEBUG)
logging.debug('this is debug')
logging.info('this is info')
logging.warning('warn message')
logging.error('error message')
logging.critical('critical message')

msg =  'requested %d vehicles, but could only find %d spawn points'
logging.basicConfig(format='%(levelname)s:%(filename)s: %(message)s', level=logging.INFO)
logging.warning(msg, 10, 20)

logging.basicConfig(filename='logger.log',level=logging.DEBUG,filemode='w')
logging.debug('this is debug')
logging.warning('this is warnining')

'''
logging 四大组件
    Loggers 提供应用程序可直接使用的接口
    Handlers 发送日志到适当的目的地
    Filters 提供了过滤日志信息的方法
    Formatters 指定日志显示格式
'''
logger = logging.getLogger("test.conf")   #创建一个logger,默认为root logger
logger1 = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)   #设置全局log级别为debug。注意全局的优先级最高

hterm =  logging.StreamHandler()    #创建一个终端输出的handler,设置级别为error
hterm.setLevel(logging.ERROR)

hfile = logging.FileHandler("access.log")    #创建一个文件记录日志的handler,设置级别为info
hfile.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   #创建一个全局的日志格式

hterm.setFormatter(formatter)   #将日志格式应用到终端handler
hfile.setFormatter(formatter)   #将日志格式应用到文件handler


logger.addHandler(hterm)    #将终端handler添加到logger
logger.addHandler(hfile)    #将文件handler添加到logger


#定义日志msg,注意此时是logger,不是logging了
logger.debug("User %s is loging" % 'jeck')
logger.info("User %s attempted wrong password" % 'fuzj')
logger.warning("user %s attempted wrong password more than 3 times" % 'mary')
logger.error("select db is timeout")
logger.critical("server is down")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('logging.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.debug('this is in logger, not in logging')

#进阶版本
#创建logger
logger = logging.getLogger()      #创建默认logger
logger1 = logging.getLogger("testlog")     #创建一个名为testlog的logger实例logger1
logger2 = logging.getLogger("testlog1")     #创建一个名为testlog1的logger实例logger2
logger3 = logging.getLogger("testlog.child")   #创建一个testlog子实例logger3


#设置logger的日志级别
logger1.setLevel(logging.DEBUG)     #将logger1日志级别设置为debug
logger2.setLevel(logging.INFO)     #将logger1日志级别设置为info
logger3.setLevel(logging.ERROR)     #将logger1日志级别设置为warning


#创建handler
hterm = logging.StreamHandler()     #输出到终端
hfile = logging.FileHandler("test.log")  #输出到文件

#定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#将日志格式应用到handler
hterm.setFormatter(formatter)
hfile.setFormatter(formatter)

#给logger添加handler
logger.addHandler(hterm)
logger.addHandler(hfile)

logger1.addHandler(hterm)
logger1.addHandler(hfile)

logger2.addHandler(hterm)
logger2.addHandler(hfile)

logger3.addHandler(hterm)
logger3.addHandler(hfile)


#记录日志信息

logger.debug('logger debug message')
logger.info('logger info message')
logger.warning('logger warning message')
logger.error('logger error message')
logger.critical('logger critical message')

logger1.debug('logger1 debug message')
logger1.info('logger1 info message')
logger1.warning('logger1 warning message')
logger1.error('logger1 error message')
logger1.critical('logger1 critical message')

logger2.debug('logger2 debug message')
logger2.info('logger2 info message')
logger2.warning('logger2 warning message')
logger2.error('logger2 error message')
logger2.critical('logger2 critical message')

logger3.debug('logger3 debug message')
logger3.info('logger3 info message')
logger3.warning('logger3 warning message')
logger3.error('logger3 error message')
logger3.critical('logger3 critical message')

import numpy as np

a = np.arange(24).reshape(2,3,4)
print(a)
print(a.shape)
