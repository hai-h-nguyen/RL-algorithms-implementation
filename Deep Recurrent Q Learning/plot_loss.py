import pickle
from matplotlib import pyplot as plt
import numpy as np
from utils import rolling_average

with open('loss.pkl', 'rb') as fp:
    # read the data as binary data stream
    losses_list = pickle.load(fp)
    raw = np.array(losses_list)
    smooth = rolling_average(raw, window_size=100)

plt.plot(smooth, 'r')
plt.show()
