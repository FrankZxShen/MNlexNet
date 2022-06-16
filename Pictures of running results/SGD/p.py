from matplotlib import pyplot as plt
import pandas as pd
from torch import adaptive_max_pool1d

adam = pd.read_csv('0.001.csv')
rms = pd.read_csv('0.05.csv')
sgd = pd.read_csv('1000.csv')
a = pd.read_csv('0.05w.csv')

plt.figure()
plt.xlabel("epoch")
plt.title("learn_rate-optimizer=SGD")
# plt.plot(sgd.iloc[0:99, 2], color="red", linewidth=2,
#          label="1000epoch_lr=0.05/weight_decay=0.005")
plt.plot(adam.iloc[0:99, 2], color="green", linewidth=2, label="lr=0.001")
plt.plot(rms.iloc[0:99, 2], color="blue", linewidth=2, label="lr=0.05")
plt.plot(a.iloc[0:99, 2], color="yellow", linewidth=2,
         label="lr=0.05/weight_decay=0.005")

plt.legend()
# plt.show()
plt.savefig('./acc2.png')
