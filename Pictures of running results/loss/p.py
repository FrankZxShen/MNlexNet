from matplotlib import pyplot as plt
import pandas as pd
from torch import adaptive_max_pool1d

adam = pd.read_csv('trainlossa.csv')
rms = pd.read_csv('trainloss.csv')
# sgd = pd.read_csv('sgd.csv')

plt.figure()
plt.xlabel("epoch")
plt.title("Ours-AlexNet(Loss)")
plt.plot(adam.iloc[0:30, 2], color="red", linewidth=2, label="Ours")
plt.plot(rms.iloc[0:30, 2], color="blue", linewidth=2, label="AlexNet")
# plt.plot(sgd.iloc[0:99, 2], color="green", linewidth=2, label="SGD")
plt.legend()
# plt.show()
plt.savefig('./t/acc2.png')
