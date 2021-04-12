'''
Author: Ful Chou
Date: 2021-03-09 16:51:47
LastEditors: Ful Chou
LastEditTime: 2021-03-22 11:26:14
FilePath: /RL-Adventure-2/server.py
Description: What this document does
'''

import socket
import json
import threading
import time

ip = '192.168.199.128'
port = 12525
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.bind((ip,port))
sock.listen(1)

print('server is runing')

# dict = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}
# list = ['1','2','3']
# data = json.dumps(list)
#data = json.dumps(dict)


#  定义一个接受和发送数据的函数，5次发送后即在服务器端强行关闭连接。
def tcplink(sock,addr):
    print("Accept the new connection from %s:%s"%addr)
    data = sock.recv(90000).decode('utf-8')
    # data = json.loads(data)
    
    print('data_env_status: ',data)
    
    
    list = ['1','2','3']
    data_parameter = json.dumps(list)
    # sock.send(b"Welcome!Here is the Sever192.168.199.128 macos .")
    sock.send(data_parameter.encode('utf-8'))
    n=int(10)
    while n>0:
        n-=1
    sock.close()
    print('Connection from %s is close.'%addr[0])
    

while True:
    s, addr = sock.accept()
    print(s, addr)
    try:
        print('现有的连接线程数:',threading.active_count())
        t = threading.Thread(target=tcplink,args=(s,addr))
        tname = t.name
        print('Now {:s} is running for connection from {:s}'.format(tname,addr[0]))
        t.start()
    except:
        print('server is block dead')


        
    

