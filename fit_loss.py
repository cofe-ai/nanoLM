from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt 

# model widths
shapes = [128, 256, 384, 512, 640, 768,  896, 1024]
# number of parameters in millions
x = [8.82,22.36,40.61,63.59,91.28,123.69,160.82,202.67] 
x_llama = [9.59,25.47,47.64,76.10,110.85,151.90,199.24,252.86]
# training loss 
train_loss = [4.52,4.25,4.16,4.10,4.04,4.01,4.00,3.98]

train_loss_wo_up = [4.73,4.48,4.50,4.42,4.36,4.33,4.29,4.31]
#train_loss2 = [4.5642,4.4974,4.3924,4.3545,4.2945,4.2545,4.2294,4.1947,4.0247,3.947]
train_loss_20k = [4.45,4.2,4.05,3.94,3.9,3.87,3.85,3.84]

train_loss_20k_llama = [4.49,4.31,4.24,4.20,4.18,4.17,4.11,4.10]



y = np.array(train_loss)
y2 = np.array(train_loss_wo_up)
y3 = np.array(train_loss_20k)
x_sample = range(int(min(x)) - 1,int(max(x)) + 5,1)
x_llama_sample=range(int(min(x_llama)) - 1,int(max(x_llama)) + 5,1)
y4 = np.array(train_loss_20k_llama)

# power law for fitting
def func(x, a, b, c):
    print(f'the coefficients a,b,c is {a,b,c}')
    return a * np.power(x, b) + c

def curve_fit_one_line(x, y, num_pred, x_sample):
    popt, pcov = curve_fit(func, x[:total_point-num_pred], y[:total_point-num_pred], p0=[1, -1, 3], maxfev=5000)
    a = popt[0] 
    b = popt[1]
    c = popt[2]
    print(a, b, c)
    print(np.sqrt(np.diag(pcov)))
    yvals = func(x_sample, a, b, c)
    return popt, np.sqrt(np.diag(pcov)), yvals

# Number of models used for prediction. Results for prediction are not used in fitting curves
num_pred=1
total_point = 8
popt, perr, yvals = curve_fit_one_line(x, y, num_pred, x_sample)
popt2, perr2, yvals2 = curve_fit_one_line(x, y3, num_pred, x_sample)
popt3, perr3, yvals3 = curve_fit_one_line(x, y4, num_pred, x_llama_sample)

fig, ax = plt.subplots()

plot1 = ax.scatter(x[:total_point-num_pred], y[:total_point-num_pred], s=60, c='royalblue')

plot1 = ax.scatter(x[total_point-num_pred:], y[total_point-num_pred:], c='red', marker="*",s=200)

plot2 = ax.plot(x_sample, yvals, 'royalblue', ls="--", label='GPT Fitted Curve with $\mu$P@7k',linewidth=3)

plot1 = ax.scatter(x[:total_point-num_pred], y3[:total_point-num_pred], s=60, c='green')
plot1 = ax.scatter(x_llama[:total_point-num_pred], y4[:total_point-num_pred], s=60, c='purple')

plot1 = ax.scatter(x[total_point-num_pred:], y3[total_point-num_pred:], c='red', marker="*",s=200)
plot1 = ax.scatter(x_llama[total_point-num_pred:], y4[total_point-num_pred:], c='red', marker="*",s=200)

plot3 = ax.plot(x_sample, yvals2, 'green', ls="--", label='GPT Fitted Curve with $\mu$P@20k',linewidth=3)


last_x = x[-1]  # 
last_y_pred = yvals2[x_sample.index(int(last_x))]  
print(f"For x = {last_x}, predicted loss = {last_y_pred}")
last_x = x_llama[-1] 
last_y_pred = yvals3[x_llama_sample.index(int(last_x))]  
print(f"For x = {last_x}, predicted loss = {last_y_pred}")


plot4 = ax.plot(x_llama_sample, yvals3, 'purple', ls="--", label='LLAMA Fitted Curve with $\mu$P@20k',linewidth=3)

plot1 = ax.scatter(x, y2 , c='gold', marker="x",s=60, label='GPT without $\mu$P@7k')


ax.set_xticks(x)
ax.grid(True, axis='both', linestyle='--', linewidth=0.7, alpha=0.7)


selected_ticks = [x[0], x[len(x) // 2],x[-2], x[-1]]
selected_labels = [x[0], x[len(x) // 2],x[-2], x[-1]]


ax.set_xticklabels(['' if tick not in selected_ticks else label for tick, label in zip(x, x)])
ax.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=15)
ax.set_xlabel("Model Size / M", fontsize=25)
ax.set_ylabel("Train Loss", fontsize =25)

plt.show()