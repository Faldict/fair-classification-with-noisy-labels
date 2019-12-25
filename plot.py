import matplotlib.pyplot as plt

plt.style.use('seaborn')

violations = []
clean_train, clean_test = [], []
corruption_train, corruption_test = [], []
proxy_fairness_train, proxy_fairness_test = [], []
lnl_train, lnl_test = [], []

with open('logs/result.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        values = line.split('\t')
        violations.append(float(values[0]))
        clean_train.append(float(values[1]))
        clean_test.append(float(values[2]))
        corruption_train.append(float(values[5]))
        corruption_test.append(float(values[6]))
        proxy_fairness_train.append(float(values[3]))
        proxy_fairness_test.append(float(values[4]))
        # lnl_train.append(float(values[7]))
        # lnl_test.append(float(values[8]))

plt.plot(violations, clean_train, label='clean data')
plt.plot(violations, corruption_train, label='corrupted data')
plt.plot(violations, proxy_fairness_train, label='surrogated fairness constraint')
# plt.plot(violations, lnl_train, label="Learning with Noisy Labels")
plt.xlabel('Fairness Violations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train')
plt.show()

plt.clf()
plt.plot(violations, clean_test, label='clean data')
plt.plot(violations, corruption_test, label='corrupted data')
plt.plot(violations, proxy_fairness_test, label='surrogated fairness constraint')
# plt.plot(violations, lnl_test, label='Learning with Noisy Labels')
plt.xlabel('Fairness Violation')
plt.ylabel('Accuracy')
plt.title('Test')
plt.legend()
plt.show()