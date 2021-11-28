from scipy.stats import weibull_min
import math
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib


data = [225, 171, 198, 189, 189, 135, 162, 135, 117, 162]

def weibull_log_likelihood(data, lambda_hat, k_hat):
    accum = -len(data)*math.log(k_hat/lambda_hat)
    for datum in data:
        accum -= (k_hat -1)*math.log(datum/lambda_hat) - (datum/lambda_hat)**k_hat

    return accum

def weibull_ll_derivative(data, lambda_hat, k_hat):
    accum = -len(data)/k_hat
    # k_hat partial
    for datum in data:
        accum += -math.log(datum/lambda_hat) 
        tmp = (datum/lambda_hat)**k_hat
        tmp *= math.log(datum/lambda_hat)
        
        accum +=tmp

    k_hat_prime = accum

    accum = +len(data)/lambda_hat
    # lambda_hat partial
    for datum in data:
        accum += (k_hat-1)*(1/lambda_hat)
        tmp = (datum**k_hat)*k_hat
        tmp *= lambda_hat**(-(k_hat+1))
        accum -= tmp


    lambda_hat_prime = accum

    return lambda_hat_prime, k_hat_prime


def print_stats(step, old_lambda_hat, old_k_hat, data, old_lambda_hat_prime, old_k_hat_prime, old_norm, grad_clipped, new_norm, lambda_hat_prime, k_hat_prime,
            no_change_iters, lambda_hat, k_hat, lr):
    print(f"step {step}")
    print(f"lambda_hat: {old_lambda_hat:.6}")
    print(f"k_hat: {old_k_hat:.6}")
    print(f"Log likelihood: {weibull_log_likelihood(data, old_lambda_hat, old_k_hat):.2f}")
    print(f"lambda_hat derivative: {old_lambda_hat_prime}")
    print(f"k_hat derivative: {old_k_hat_prime}")
    print("orig norm", old_norm)
    if grad_clipped:
        print("new norm", new_norm)
        print(f"new lambda_hat derivative: {lambda_hat_prime}")
        print(f"new k_hat derivative: {k_hat_prime}")

    print(f"no_change_iters: {no_change_iters}")
    print(f"new lambda_hat {lambda_hat:.6}")
    print(f"new k_hat {k_hat:.6}")
    print(f"new lr: {lr:}\n")



lambda_hat, k_hat = 1.0, 1.0
decay_base= 1-1e-10
lr = .5
step = 0

lambda_hat_guesses = [lambda_hat]
k_hat_guesses = [k_hat]

# Step wise gradient descent
no_change_iters = 0
while no_change_iters < 4:
    step += 1
    lambda_hat_prime, k_hat_prime = weibull_ll_derivative(data, lambda_hat, k_hat)

    # compute norm of the gradient
    primes = np.array([lambda_hat_prime, k_hat_prime])
    orig_norm = np.linalg.norm(primes)

    # store values in case they change
    old_lambda_hat_prime = lambda_hat_prime
    old_k_hat_prime = k_hat_prime

    # check for out of control gradient 
    if orig_norm > 10:
        # "clip" gradient norm
        primes = primes * 10/np.linalg.norm(primes)

        # store re-scaled gradients
        lambda_hat_prime = primes[0]
        k_hat_prime = primes[1]
        new_norm = np.linalg.norm(primes)
    else:
        new_norm = orig_norm

    old_lambda_hat = lambda_hat

    # gradient descent update
    lambda_hat = lambda_hat - lr*lambda_hat_prime

    # enforce parameter limit
    lambda_hat = max(0+1e-2, lambda_hat)
    lambda_hat_guesses.append(lambda_hat)
    old_k_hat = k_hat

    # gradient descent update
    k_hat = k_hat - lr*k_hat_prime

    # enforce parameter limit
    k_hat = max(0+1e-2, k_hat)
    k_hat_guesses.append(k_hat)

    # decay learning rate
    lr = lr *(decay_base)**step

    print_stats(step, old_lambda_hat, old_k_hat, data, old_lambda_hat_prime, old_k_hat_prime, orig_norm, orig_norm > 10, new_norm, lambda_hat_prime, k_hat_prime,
            no_change_iters, lambda_hat, k_hat, lr)

    if abs(old_lambda_hat-lambda_hat) < 1e-3 and abs(old_k_hat-k_hat) < 1e-3:
        no_change_iters +=1
    else:
        no_change_iters = 0

    if step > 50000: 
        break

# Change font size
font = {'size'   : 22}
matplotlib.rc('font', **font)

# make plots
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(16, 16))
ax1.plot([i+1 for i in range(len(k_hat_guesses))], k_hat_guesses)
ax1.set_title(f"K Hat Trajectory")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("K Hat Value")


ax2.plot([i+1 for i in range(len(lambda_hat_guesses))], lambda_hat_guesses)
ax2.set_title(f"Lambda Hat Trajectory")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Lambda Hat Value")

ax3.hist(data, density=True)
x = np.linspace(weibull_min.ppf(0.01, k_hat, loc=0, scale=lambda_hat),
                weibull_min.ppf(0.99, k_hat, loc=0, scale=lambda_hat), 100)
ax3.plot(x, weibull_min.pdf(x, k_hat, loc=0, scale=lambda_hat),
       'r-', lw=5, alpha=0.6, label='weibull_min pdf')
ax3.set_title(f"K Hat: {k_hat:.3f}, Lambda Hat: {lambda_hat:.3f}")
ax3.set_xlabel("Data Value")
ax3.set_ylabel("Density")

plt.tight_layout()
fig.savefig("weibull_estimation.png")





