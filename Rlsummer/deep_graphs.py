import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def gas_graph():
    x = list(np.arange(350, 800, 100))
    print(x)
    for p in x:
        format(p, ',')
    y = [2 * i for i in x]
    print(y)
    plt.plot(x, y, 'o-', color="red")
    for i, j in zip(x, y):
        plt.text(i+10, j - 50, str(j))
    plt.text(350, 1400, " with $2 fee per transaction ")
    plt.xlabel("Number of Transactions at ATM")
    plt.ylabel("Profit earned in $ amount")
    plt.title("Projections for profit earned through ATM Transactions")
    plt.show()


gas_graph()
