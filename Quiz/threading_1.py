import threading
import time

def addHarmonicSeries(n):
    sum = 0.0
    for m in range(1, n):
        sum = sum + 1.0/m
        print('{:5d} {:12.6f}'.format(m, sum))

start_time = time.clock()
no_thread = 10000
for i in range(no_thread):
        t = threading.Thread(target = addHarmonicSeries, args = (i,))
        t.start()
        print("\n-->  "+t.getName() + "\n")
        #time.sleep(1)

print("\n The main stread stops after {}".format(time.clock()-start_time) + " seconds")








