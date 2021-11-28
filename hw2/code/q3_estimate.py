from scipy.stats import cauchy
import math
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt


data = [7.52, 9.92, 9.52, 21.97, 8.39, 8.09, 9.22, 9.37, 7.33, 15.32, 1.08, 8.51, 17.73, 11.20, 8.33, 10.83, 12.40, 14.49, 9.44, 3.67]


theta_hat = 0
def cauchy_log_likelihood(data, theta_hat):
    accum = 0
    for datum in data:
        accum += +math.log(math.pi) + math.log(1+(datum-theta_hat)**2)

    return accum

def cauchy_ll_derivative(data, theta_hat):
    accum = 0
    for datum in data:
        accum += -2*(datum-theta_hat)/(1+(datum-theta_hat)**2)

    return accum

def newton_raphson_lr(data=None, theta_hat=None, **kwargs):
    accum = 0
    for datum in data:
        accum += 2*(1/(1+(datum-theta_hat)**2) +(datum-theta_hat)/((1+(datum-theta_hat)**2)**2))

    return 1/accum


def print_stats(step, old_theta_hat, data, delta, lr, no_change_iters, theta_hat):
    print(f"step {step}")
    print(f"theta_hat: {old_theta_hat:.6}")
    print(f"Log likelihood: {cauchy_log_likelihood(data, old_theta_hat):.2f}")
    print(f"derivative: {delta:.2f}")
    print(f"lr: {lr:.5}")
    print(f"no_change_iters: {no_change_iters}")
    print(f"new theta_hat {theta_hat:.6}")
    print(f"new Log likelihood: {cauchy_log_likelihood(data, theta_hat):.2f}")
    print(f"new lr: {lr:.5}\n")


def lr_decay(lr=None, decay_base=None, step=None, **kwargs):
    return lr *(decay_base)**step

def get_full_set(data):
    return data

def sample_uniform(data):
    samp_idx = random.randint(0, len(data)-1) #second param is inclusive 
    return data[samp_idx:samp_idx+1] # return as list rather than float

def gradient_descent_update(param, lr, delta):
    return param - lr*delta

parser = argparse.ArgumentParser(description='Estimate theta')
parser.add_argument('--optim', default="sgd",
                    choices=["sgd", "swgd", "nr"],
                    help='optimizer to use')

args = parser.parse_args()

theta_hat = 0.
decay_base= 1-1e-3
lr = .5
step = 0
stability_iter_limit = 4
stability_threshold = 1e-5

guesses = [theta_hat]
#Step-wise
if args.optim == "swgd":
    lr_update = lr_decay
    data_sampler = get_full_set

elif args.optim == "nr":
    lr_update = newton_raphson_lr
    data_sampler = get_full_set

elif args.optim == "sgd":
    lr_update = lr_decay
    data_sampler = sample_uniform



no_change_iters = 0
while no_change_iters < stability_iter_limit: # finish when theta hat stops changing much
    step += 1
    samp_data = data_sampler(data)
    delta = cauchy_ll_derivative(samp_data, theta_hat)
    old_theta_hat = theta_hat

    # gradient descent update
    theta_hat = theta_hat - lr*delta

    guesses.append(theta_hat)
    lr = lr_update(lr=lr, decay_base=decay_base, step=step, data=data, theta_hat=theta_hat)
    print_stats(step, old_theta_hat, data, delta, lr, no_change_iters, theta_hat)
    if abs(old_theta_hat-theta_hat) < stability_threshold:
        no_change_iters +=1
    else:
        no_change_iters = 0


# Make plots
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot([i+1 for i in range(len(guesses))], guesses)
ax1.set_title(f"Theta Hat Trajectory")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Theta Hat Value")


ax2.hist(data, density=True)
x = np.linspace(cauchy.ppf(0.01, theta_hat),
                cauchy.ppf(0.99, theta_hat), 100)
ax2.plot(x, cauchy.pdf(x, theta_hat),
       'r-', lw=5, alpha=0.6, label='cauchy pdf')
ax2.set_title(f"Theta hat: {theta_hat:.3}")
ax2.set_xlabel("Data Value")
ax2.set_ylabel("Density")
plt.tight_layout()
fig.savefig("cauchy_"+args.optim+".png")

