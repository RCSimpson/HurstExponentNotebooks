import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def RSI(X, window, graph):
    
    N = len(X)
    UpDiff = np.zeros(N - 1)
    DownDiff = np.zeros(N - 1)

    for i in np.arange(N-1):
        difference = X[i] - X[i-1]
        if difference > 0:
            UpDiff[i] = abs(difference)
            DownDiff[i] = 0
        elif difference < 0:
            UpDiff[i] = 0
            DownDiff[i] = abs(difference)
        elif difference == 0:
            UpDiff[i] = 0
            DownDiff[0] = 0

    d1 = pd.DataFrame(UpDiff)
    d2 = pd.DataFrame(DownDiff)
    SMMAUp = d1.ewm(span= window).mean().to_numpy()
    SMMADown = d2.ewm(span = window).mean().to_numpy()

    RS = SMMAUp/SMMADown
    RSIValue = 100 - 100/(1 + RS.flatten())

    if graph == True:
        plt.figure(figsize=(10, 8))
        plt.plot(RSIValue, label='RSI')
        plt.xlabel('Time')
        plt.ylabel('RSI')
        plt.title('Relative Strengh Index')
        plt.legend()
        plt.show()

    return RSIValue

X =  np.cumsum(np.random.randn(1000))
RsIndex = RSI(X, 14, graph=True)