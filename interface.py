import tkinter
#from tkinter import*
import time
import shutil
from multiprocessing import Process, Queue

class interface():
    def __init__(self, q):
        # 创建布局
        self.root = tkinter.Tk()
        self.root.title("Sound Detector")
        self.fmleft = tkinter.Frame(self.root)
        self.fmleft.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=tkinter.YES)
        self.fmright = tkinter.Frame(self.root)
        self.fmright.pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=tkinter.YES)

        # 创建初始画布
        self.canvas = tkinter.Canvas(self.fmleft,width=500,height=500)
        self.canvas.create_oval(0, 0, 500, 500)
        self.canvas.create_oval(50, 50, 450, 450)
        self.canvas.create_oval(100, 100, 400, 400)
        self.canvas.create_oval(150, 150, 350, 350)
        self.canvas.create_oval(200, 200, 300, 300)
        self.canvas.create_line(0, 250, 500, 250)
        self.canvas.create_line(250, 0, 250, 500)
        self.canvas.pack()

        # 创建消息框
        self.text = tkinter.Text(self.fmright,width=20,height=50)
        self.text.pack(anchor="n")

        self.scroll = tkinter.Scrollbar()
        self.scroll.pack(side=tkinter.RIGHT,fill=tkinter.Y)

        #绑定滚动条和消息框
        self.scroll.config(command=self.text.yview)
        
        # 绑定线程变量传递函数
        self.notify_queue = q

    def method(self, time, x, y):
        #text.insert(INSERT,"第"+str(count)+"条消息\n")

        self.canvas.delete(tkinter.ALL)
        self.canvas.create_oval(0, 0, 500, 500)
        self.canvas.create_oval(50, 50, 450, 450)
        self.canvas.create_oval(100, 100, 400, 400)
        self.canvas.create_oval(150, 150, 350, 350)
        self.canvas.create_oval(200, 200, 300, 300)
        self.canvas.create_line(0, 250, 500, 250)
        self.canvas.create_line(250, 0, 250, 500)
        
        self.message = str(time) + "s, voice in (" + str(x) + ", " + str(y) + ").\n"
        
        size = 8
        
        if x < size: x = size
        elif x > 500 - size: x = 500 - size
        
        if y < size: y = size
        elif y > 500 - size: y = 500 - size
        
        self.canvas.create_oval(x-size, y-size, x+size, y+size, fill = "red")
        self.text.insert(tkinter.INSERT, self.message)
    
    
    def mainloop(self):
        '''
        # 用于测试   
        btn = Button(fmleft, text = "click me", command = method)
        btn.pack()
        '''
        self.helloworld()
            
        self.root.mainloop()
        
    def helloworld(self):
        self.root.after(1000, self.helloworld)
        if not self.notify_queue.empty():
            value = self.notify_queue.get()
            self.method(value[0], value[1], value[2])        


def interface_fetch(i, q):
    target = interface(q)
    target.mainloop()
            
if __name__ == "__main__":
    q = Queue()
    count = 50
    inter = Process(target=interface_fetch, args=(0, q))
    inter.start()
    while count < 180:
        count += 1
        #method(count, count, count)
        q.put([count, count, count])
        print(count)
        time.sleep(1)
        
'''
1m = 50单位
1cm = 0.5单位
'''