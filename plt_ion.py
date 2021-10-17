import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(0,30,0.1)
plt.figure()
plt.ion()
for i in range(100):
    plt.ylim(-10,10)
    plt.plot(t1,0.1*i*np.sin(t1))
    plt.pause(0.1)
    plt.clf()
plt.ioff()

# plt.ion()
# plt.figure()
# plt.pause()
# plt.clf()
# plt.ioff()