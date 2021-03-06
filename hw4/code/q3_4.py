from pydataset import data
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import dirichlet
import pandas as pd
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

LOCATION_ONLY = True
SAVEPLOTS = True
STOCHASTIC = True
FREEZE_MU_ITERS = 20
MIN_SIGMA = .001
SAVEDIR = "stochastic_" if STOCHASTIC else  "regular_"
SAVEDIR += "loc_only_plots" if LOCATION_ONLY else "loc_scale_plots" 
CONV_CRITERIA = {'tol':1e-1, 'consec':1} if STOCHASTIC else {'tol':1e-3, 'consec':10}
EPSILON = 1e-14

if LOCATION_ONLY:
    ks = [4, 6, 8, 11, 15, 20]
else:
    ks = [3, 4, 5, 6, 7, 8]

df = data("galaxies")/1000
pd.set_option("display.max_rows", 200)

def make_mixture_pdf(pis, mus, sigmas):
    def mixture_pdf(x, silent=True):
        # assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(pis, np.ndarray)
        p = 0.0
        for j in range(pis.shape[1]):
            if not LOCATION_ONLY:
                tmp = pis[0,j]*norm.pdf(x, loc=mus[0,j], scale=(sigmas[0,j]+EPSILON))
                p += tmp
                if not silent:
                    print(j, x,  f"pi: {pis[0,j]}, mu: {mus[0,j]}, p: {tmp}")

            else: 
                tmp = pis[0,j]*norm.pdf(x, loc=mus[0,j], scale=(sigmas[0,0]+EPSILON))
                p += tmp

        return p
    return mixture_pdf

def compute_LB(k, data, z_post, new_pis, new_mus, new_sigmas):
    # Compute new joint z,y dist'n
    # zy_joint = compute_zy_joint_dist(k, data, new_pis, new_mus, new_sigmas)
    # # z_marg dims: [1, k]
    # tmp = z_post*(np.log(zy_joint+EPSILON))# - np.log(z_post+EPSILON))
    # return tmp.sum()
    return compute_Likelihood(data, new_pis, new_mus, new_sigmas)

def compute_Likelihood(data, pis, mus, sigmas):
    mix_pdf = make_mixture_pdf(pis, mus, sigmas)
    return np.log(mix_pdf(data)).sum()
    # print(ps)

def plot_data(data, bins=20):
    plt.hist(df[:,0], density=True, bins=bins)

def plot_density_overlay(pis, mus, sigmas, minimum, maximum, it, silent=True):
    mix_pdf = make_mixture_pdf(pis, mus, sigmas)
    bins = 20
    xs = np.linspace(minimum, maximum, 1000)
    # ys = np.array([mix_pdf(xs[i], silent=silent and (i%100==0)) for i in range(xs.shape[0])])
    ys = np.array([mix_pdf(xs[i], silent=True) for i in range(xs.shape[0])])
    plt.plot(xs,ys, label=f"iter={it}")


def init_mus(k, maximum=35, minimum=0):
    # mus = norm.rvs(loc=x_bar, scale=x_bar_sd, size=k)
    # mus = mus[np.newaxis,:]
    mus = np.random.random((1,k))*(maximum-minimum)

    return mus

def init_sigma_2s(n,k):
    # chi_2s = chi2.rvs(n-1, size=k)
    # sigma_2s = chi_2s*s_2/(n-1)
    # sigma_2s = sigma_2s[np.newaxis,:]

    sigma_2s = np.ones((1,k))*1000

    # sigma_2s dims: [1,k]
    if LOCATION_ONLY:
        sigma_2s = sigma_2s[:,:1]
        # sigma_2s dims: [1,1]

    return sigma_2s

def init_pis(k):
    # alphas = np.empty(k)
    # alphas.fill(1)
    # pis = dirichlet.rvs(alphas, size=1) 

    pis = np.ones((1,k))/k
    # pis dims: [1,k]
    return pis

def compute_zy_joint_dist(k, data, pis, mus, sigmas):
    assert len(data.shape) == 1
    assert pis.shape == (1,k)
    assert mus.shape == (1,k)
    # Create joint p(z,y|...) matrix
    n = data.shape[0]
    zy_joint = np.zeros((n, k))
    # zy_joint dims: [n, k]

    # Fill matrix with f(y_i|zai_j)s.  
    for j in range(k):
        if not LOCATION_ONLY:
            tmp = norm.pdf(data, loc=mus[0,j], scale=(sigmas[0,j]+EPSILON))
        else:
            tmp = norm.pdf(data, loc=mus[0,j], scale=(sigmas[0,0]+EPSILON))
        zy_joint[:,j] = tmp            

    # Multiply with f(z_ij|theta)s 
    zy_joint = zy_joint*pis
    return zy_joint

def sample_z_idxs(z_dist):
    n, k = z_dist.shape
    z_idxs = [np.random.choice(k, p=z_dist[i]) for i in range(n) ]
    return z_idxs



def stochastic_update(z_post, zy_joint, data, pis, mus, sigmas):
    n, k = z_post.shape

    # sample z_idx for each data point
    z_idxs = sample_z_idxs(z_post)
    # z_idxs type: list, len: n

    # Map z_idx to data index
    z_map = defaultdict(list)

    # Record z_idx for each
    z_df = pd.DataFrame().from_dict({"z":z_idxs, "data":data[:,0].tolist()})

    # Get the unique z_idxs present and their counts 
    # NOTE: there's a chance an index wasn't sampled
    zs, z_counts = np.unique(z_idxs, return_counts=True)

    new_pis = np.zeros(pis.shape)
    new_mus = np.zeros(mus.shape)
    # Iterate each z_idx/centroid
    for z_val, count in zip(zs, z_counts):
        new_pis[0,z_val] = count/n
        # percent of times z_val sampled

        new_mus[0,z_val] = z_df.loc[z_df["z"]==z_val, "data"].sum()/count
        # average of datapoints generated by z_val


    new_sigma_2s = np.zeros(sigmas.shape)
    for z_val, count in zip(zs, z_counts):
        
        if LOCATION_ONLY:
            tmp = ((z_df.loc[z_df["z"]==z_val, "data"]-new_mus[0,z_val])**2).sum()/count
            # average distance of all datapoints genrated by z_val from its new mean

            new_sigma_2s[0,0] += tmp.sum()/k
            # average all variances

        else:
            new_sigma_2s[0,z_val] = ((z_df.loc[z_df["z"]==z_val, "data"]-new_mus[0,z_val])**2).sum()/count

    new_sigmas = np.sqrt(new_sigma_2s)

    dim0 = [0 for i in range(n)]
    new_q = np.log(zy_joint[dim0, z_idxs]+EPSILON).sum()
    return new_pis, new_mus, new_sigmas, new_q

        



def normal_update(z_post, data):
    # data dims: [n,1]
    n, k = z_post.shape
    assert data.shape == (n, 1)

    weights = z_post/ z_post.sum(axis=0, keepdims=True)
    # weights dims: [n, k]
    # NOTE: Each i,j weight can be interpreted as the contribution of a centroid, j, to a datapoint, i,
    #       as a fraction of jth's centroid total contribution to all datapoints
    #       I.e. how much each data point "matters" to a centroid

    # Pi's
    new_pis = z_post.sum(axis=0, keepdims=True)/n
    # Average contribution of each centroid to all data points 
    # new_pis dims: [1,k]
   
    # Mu's
    if it < FREEZE_MU_ITERS:
        new_mus = mus
    else:

        new_mus = (weights*df).sum(axis=0, keepdims=True)
        # Weighted contribution of each data point to means of new centroids
    # new_mus dims: [1,k]

    # Sigma^2's
    new_sigma_2s = (weights*((df- new_mus)**2)).sum(axis=0, keepdims=True)
    # Weighted squared distance from each centroid mean

    # new_sigma_2s dims: [1,k]
    if LOCATION_ONLY:
        new_sigma_2s = (z_post*((df- new_mus)**2)).sum(axis=1, keepdims=True).sum(axis=0, keepdims=True)/n

    new_sigmas = np.sqrt(new_sigma_2s)
        
    new_sigmas = np.clip(new_sigmas, a_min=MIN_SIGMA, a_max=None)

    return new_pis, new_mus, new_sigmas

x_range = df.max() - df.min()
n = df.shape[0]

x_bar = df.mean()
s_2 = (((df-x_bar)**2).sum() / (n-1)).tolist()[0]
x_bar_var = s_2 / n
x_bar_sd = math.sqrt(x_bar_var)

s_2_var = (2*s_2**2)/(n-1)
s = math.sqrt(s_2)
s_2_sd = math.sqrt(s_2_var)

print(f"x_bar: {x_bar.tolist()[0]}")
print(f"x_bar_var: {x_bar_var}")
print(f"x_bar_sd: {x_bar_sd}")
print(f"s: {s}")
print(f"s_2: {s_2}")
print(f"s_2_var: {s_2_var}")
print(f"s_2_sd: {s_2_sd}")

df = df.to_numpy()
# df (i.e. y) dims: [n, 1]
criterion_data = {"aic":[], "bic":[], "k":[], "iters":[]}

for k in ks:
    # Initialize mus
    mus = init_mus(k)
    # mus dims: [1,k]


    # Initialize sigma^2s
    sigma_2s =  init_sigma_2s(n, k)
    sigmas = np.sqrt(sigma_2s)

    # Initialize pis
    pis = init_pis(k)
    # pis dims: [1,k]

    # Report
    print("*"*50)
    print(f"k={k}")
    print(f"Init vals:")
    print("\tmus:", mus)
    print("\tpis:", pis)
    print("\tsigs:", sigmas)

    prev_q = 100
    delta = 1
    # theta_old = np.ones((1, pis.shape[1]+mus.shape[1]+sigma_2s.shape[1]))
    # theta_old = theta_old/np.linalg.norm(theta_old)
    it = 0
    plt.clf()
    plot_data(df[:,0])
    plot_density_overlay(pis, mus, sigmas, df[:,0].min(),df[:,0].max(), "init")
    consec = 0
    while consec <= CONV_CRITERIA['consec'] or it < FREEZE_MU_ITERS+5:
        if it % 20 == 0:
            print(it,  end="\r")

        if abs(delta) < CONV_CRITERIA['tol']:
            consec += 1
        else:
            consec = 0

        print(it, f"{delta:.4f}")
        
        
        ###### E-step: First compute dist'n f(z|y,theta) 
        #       # Normal Mixture application of EM:
        #           f(z|y,theta) = f(z, y|theta)/f(y|theta) =  f(z, y|theta)/(sum_z f(y, z|theta)f(z|theta))
        #                                                   = f(y|z, theta)*f(z|theta)/sum_z f(y|z, theta)*f(z|theta)
        #               Numerator -- f(y|z, theta)*f(z|theta)
        #               Denominator --sum_z f(y|z, theta)*f(z|theta)
        #
        # Expanding the data dim. The full z is a matrix (an n by k matrix in my implementation), and z_i is now a (row) vector.
        # Here I index the rows by i. Likewise above y was a column vector, y had entry per data point. y_i is now a scalar.
        # Since this model assumes data is iid, all rows will have same dist'n and full vector dist'n is product of all rows.
        # Therefore each row is a valid dist'n and will sum to one.
        #       f(z_i|y_i,theta) = 
        #           Numerator -- f(y_i|z_i, theta)*f(z_i|theta)
        #           Denominator -- sum_j f(y_i|z_ij, theta)*f(z_ij|theta)
        #
        # Expanding the z_i dim. Here I index each column entry by j:
        #       f(z_ij|y_i,theta) = 
        #       Numerator --  f(y_i|z_ij, theta)*f(z_ij|theta) = {f(y_i| zai_j)*f(z_ij|theta)}^I(z_ij=1)
        #       Denominator --  sum_j f(y_i|z_ij, theta)*f(z_ij|theta) =  sum_j {f(y_i|zai_j, theta)*f(z_ij|theta)}^I(z_i=j)

        ## Compute numerators
        zy_joint = compute_zy_joint_dist(k, df[:,0], pis, mus, sigmas)
        # zy_joint dims: [n, k]

        # Compute denominators
        z_post = zy_joint/(zy_joint.sum(axis=1, keepdims=True))
        # z_post dims: [n, k]
        # NOTE: z_post can be interpreted as the contribution of each centroid to generating a particular data point

        ####### M-Step: Maximize  E_q(z)[log(p(y, z|theta))] = E_q(z)[L(theta) - log(p(z|y,theta))] 
        #      Use updates from taking derivatives and setting to zero (Lagrange multiplier necessary for pis to sum to 1) 
        if STOCHASTIC:
            pis, mus, sigmas, new_q = stochastic_update(z_post, zy_joint, df, pis, mus, sigmas)

        else:
            pis, mus, sigmas = normal_update(z_post, df)
            new_q = compute_LB(k, df[:,0], z_post, pis, mus, sigmas)


        
        # plot_density_overlay(pis, mus, sigmas, df[:,0].min(),df[:,0].max(), it)

        # new_q = compute_LB(k, df[:,0], z_post, pis, mus, sigmas)
        delta = new_q - prev_q          
        prev_q = new_q
        it+=1    

    p = np.prod(pis.shape) + np.prod(mus.shape) + np.prod(sigmas.shape)
    L = compute_Likelihood(df[:,0], pis,mus,sigmas) 
    aic = 2*(-L + p)
    bic = -2*L + p*math.log(n)
    print(f"New vals:")
    print("\tmus:", mus)
    print("\tpis:", pis)
    print("\tsigs:", sigmas)
    print("Iterations til convergence:", it)
    print("AIC:", aic)
    print("BIC:", bic)
    # print("delta", delta)
    plot_density_overlay(pis, mus, sigmas, df[:,0].min(),df[:,0].max(), "fin", silent=False)
    plt.ylim((0,.285))
    plt.legend()
    # plt.title(f"K:{k}")
    #, AIC:{aic:.1f}, BIC:{bic:.1f}")
    
    if SAVEPLOTS:
        plt.savefig(Path(SAVEDIR)/ f"galaxies_hist_k_{k}.png")


    criterion_data["aic"].append(aic)
    criterion_data["bic"].append(bic)
    criterion_data["k"].append(k)
    criterion_data["iters"].append(it)

def format_flts(lst):
    return [f"{flt:.1f}" for flt in lst]

def my_argmin(lst):
    return min([(i,el) for i, el in enumerate(lst)], key=lambda x: x[1])[0]

def my_formatter(lst):
    idx = my_argmin(lst)
    fmted = format_flts(lst)
    fmted[idx] += "*"
    return fmted

criterion_data["aic"] = my_formatter(criterion_data["aic"])
criterion_data["bic"] = my_formatter(criterion_data["bic"])
print(criterion_data)
crit_df = pd.DataFrame.from_dict(criterion_data)
crit_df = crit_df.set_index("k")
print(crit_df.to_latex())


        

     


