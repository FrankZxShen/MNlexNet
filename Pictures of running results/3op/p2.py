from matplotlib import pyplot as plt
import pandas as pd
from torch import adaptive_max_pool1d

adam = pd.read_csv('adam.csv')
rms = pd.read_csv('rms.csv')
sgd = pd.read_csv('sgd.csv')

plt.figure()
plt.xlabel("epoch")
plt.title("optimizer-learn_rate=0.001")
plt.plot(adam.iloc[0:99, 2], color="red", linewidth=2, label="Ours")
plt.plot(rms.iloc[0:99, 2], color="blue", linewidth=2, label="RMSprop")
plt.plot(sgd.iloc[0:99, 2], color="green", linewidth=2, label="SGD")
plt.legend()
# plt.show()
plt.savefig('./t/acc.png')
