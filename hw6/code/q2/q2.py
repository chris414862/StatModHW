import numpy as np
import random
import matplotlib.pyplot as plt
from pydataset import data
from scipy.stats import invgamma, norm, dirichlet, multivariate_normal
from scipy.special import logsumexp

# Get data
df = data('faithful')
df = df['waiting']
data = df.to_numpy()[np.newaxis,:]

# Number of components in each mixture. 
K = 5
T = data.shape[1]




# Define hyperpriors
v = df.var()
mu_0 = df.mean()
sigma2_0 = 2#mu_0/(df.shape[0]**(1/2))
kappa_0 = 1
nu_0 = 2



#### helper funcs
def get_n_jks(zs):
    """
    Collects n_jk's: # transitions from j to k
    """
    # zs dim: [1, T]
    #  contains indeces of non-zero entries of full z vectors
    ns = np.zeros((K, K))
    for k in range(K):
        # print("k:", k)
        dummy_zs = zs[:, 1:]
        zs_p1 = zs[:, :T-1]
        trns_idxs = zs_p1[:, dummy_zs[0]==k]
        trns_info = np.unique(trns_idxs, return_counts=True)
        for trns_idx, cnt in zip(*trns_info):
            ns[int(trns_idx), k] += cnt

    return ns

### Conditionals
# pis
def get_cond_pis_sampler(prior_dir_alphas):

    def cond_pis_sampler(zs):
        # zs dim: [1, T]
        #  contains indeces of non-zero entries of full z vectors
        T = zs.shape[1]
        # Collect n_jk's: # transitions from j to k
        ns = get_n_jks(zs)
        cond_alphas = np.repeat(prior_dir_alphas, K, axis=0)
        cond_alphas += ns
        cond_pis = []
        for j in range(K):
            cond_pis.append(dirichlet.rvs(cond_alphas[j], size=1))

        sampled_cond_pis = np.concatenate(cond_pis, axis=0)
        return sampled_cond_pis

    return cond_pis_sampler


def get_nig_sampler(mu_0, sigma2_0, kappa_0, nu_0):
    a_0= nu_0/2
    invg_scale=sigma2_0*nu_0/2 #Doesn't seem like /2 will neet to be included for scipy.invgamma
    def nig_sampler():
        sigma_2 = invgamma.rvs(a_0, scale=invg_scale, size=1)
        mu = norm.rvs(loc=mu_0, scale=np.sqrt(sigma_2/kappa_0), size=1)
        return  mu[0], sigma_2[0]
    return nig_sampler

def get_cond_mu_sigma2_sampler(nu_0, sigma2_0, mu_0, kappa_0):
    def cond_mu_sigma2_sampler(zs, data):
        # zs dims: [1,T]
        # data dims: [1,T}
        T = data.shape[1]
        cond_mus = np.zeros((K,1))
        cond_sigma2s = np.zeros((K, 1))
        for k in range(K):
            mask = zs[0,:] == k
            k_state_idxs = np.arange(T)[mask] 
            k_state_data = data[:, k_state_idxs]
            if mask.sum() == 0:
                k_state_mu_hat = 0
                k_state_s2 = 0
            else:
                k_state_mu_hat = k_state_data.mean()
                k_state_s2 = k_state_data.var()
            k_state_n = mask.sum()
            cond_nu_0 = nu_0 + k_state_n
            cond_kappa_0 = kappa_0 + k_state_n
            cond_mu_0 = (kappa_0*mu_0 + k_state_n*k_state_mu_hat)/(kappa_0 + k_state_n)
            cond_sigma2_0 = 1/cond_nu_0 *(nu_0*sigma2_0 + (k_state_n-1)*k_state_s2 + \
                                            ((k_state_n*kappa_0)/(k_state_n+kappa_0))*(k_state_mu_hat - mu_0)**2)
            new_mu, new_sigma2 = get_nig_sampler(cond_mu_0, cond_sigma2_0, cond_kappa_0, cond_nu_0)()
            cond_mus[k, 0] = new_mu
            cond_sigma2s[k, 0] = new_sigma2

        return cond_mus, cond_sigma2s
            
    return cond_mu_sigma2_sampler

def cond_z_sampler(s, pis, data, mus, sigma2s):
    # pis dims: [K, K], s dims: [1, K], data dims: [1,T],  mus dims: [K,1],  sigmas dims: [K,1]
    K = s.shape[1]
    T = data.shape[1]
    phis = np.zeros((K, T))
    log_pis = np.log(pis)
    normalizers = np.zeros(T)
    tmp = np.zeros((K,1))+np.log(s.T)
    tmp += np.log(norm.pdf(data[0,0],loc=mus[:,0], scale=np.sqrt(sigma2s)[:,0])[:, np.newaxis])
    phis[:,0:1] = tmp

    
    # phis dims: [K, T]
    for t in range(1,T):
        phi_t = logsumexp(phis[:,t-1:t] + log_pis, axis=0, keepdims=True).T #dims: [K,1]
        tmp = np.log(norm.pdf(data[0,t],loc=mus[:,0], scale=np.sqrt(sigma2s)[:,0]))[:,np.newaxis] #dims: [K,1]
        phi_t += tmp
        phis[:,t:t+1] = phi_t

    idxs = np.arange(K)
    final_dist = np.exp(phis[:,-1] - logsumexp(phis[:,-1]))
    zs = [np.random.choice(idxs, size=1, p=final_dist)[0]]
    for t in range(T-2, -1, -1):
        samp_dist = phis[:, t]+log_pis[:,zs[-1]]
        samp_dist = np.exp(samp_dist-logsumexp(samp_dist))
        zs.append(np.random.choice(idxs, size=1, p=samp_dist)[0])

    zs = zs[::-1]
    return np.array(zs)[np.newaxis,:]



# initialize variables
zs = np.zeros((1, T))-1
prior_dir_alphas = np.full((1,K), 1/K)
dir_prior_sampler = lambda : dirichlet.rvs(alpha=prior_dir_alphas[0], size=1)
pis = np.concatenate([dir_prior_sampler() for i in range(K)], axis=0)

# # Gibbs sampler
burnin = 100
reps = 1000
thinning = 10
mus_samples, pis_samples, sigma2s_samples, stat_dist_samples = [], [], [], []
s = dir_prior_sampler()
for i in range(reps):
    print(" ", i,end="\r")
    cond_mu_sigma2_sampler = get_cond_mu_sigma2_sampler(nu_0, sigma2_0, mu_0, kappa_0)
    cond_pis_sampler = get_cond_pis_sampler(prior_dir_alphas)
    mus, sigma2s = cond_mu_sigma2_sampler(zs, data)#[:,:T])
    pis = cond_pis_sampler(zs)
    zs = cond_z_sampler(s, pis, data, mus, sigma2s)
    mus_samples.append(mus)
    sigma2s_samples.append(sigma2s)
    pis_samples.append(pis[:,:,np.newaxis])

    # Get stationary distribution
    eig_vals, eig_vecs = np.linalg.eig(pis.T)
    stat_dist_idx = np.arange(eig_vals.shape[0])[np.isclose(eig_vals,1.)][0]
    stat_dist = eig_vecs[:,stat_dist_idx:stat_dist_idx+1]
    stat_dist = np.abs(stat_dist/stat_dist.sum())
    stat_dist_samples.append(stat_dist)


# Make tensor from all samples
mus_samples =np.concatenate(mus_samples, axis=1)
sigma2s_samples =np.concatenate(sigma2s_samples, axis=1)
pis_samples =np.concatenate(pis_samples, axis=2)

# burn in and thinning
mus_samples = mus_samples[:, burnin::thinning]
sigma2s_samples = sigma2s_samples[:, burnin::thinning]
pis_samples = pis_samples[:, :,burnin::thinning]

# Get estimates
mus_hat = mus_samples.mean(axis=1, keepdims=True)
sigma2s_hat = sigma2s_samples.mean(axis=1, keepdims=True)
pis_hat = pis_samples.mean(axis=2)

# Get stationary dist from full estimate
eig_vals, eig_vecs = np.linalg.eig(pis_hat.T)
stat_dist_idx = np.arange(eig_vals.shape[0])[np.isclose(eig_vals,1.)][0]
stat_dist_hat = eig_vecs[:,stat_dist_idx:stat_dist_idx+1]
stat_dist_hat = np.abs(stat_dist.T/stat_dist.sum())
print("stat_dist", stat_dist_hat)


# Funcs for plotting
def make_mixture_pdf(pis, mus, sigmas):
    def mixture_pdf(x, silent=True):
        # assert isinstance(x, float) or isinstance(x, int)
        assert isinstance(pis, np.ndarray)
        p = 0.0
        for j in range(pis.shape[0]):
            # if not LOCATION_ONLY:
            EPSILON = 0#1e-10
            tmp = pis[j, 0]*norm.pdf(x, loc=mus[j, 0], scale=(sigmas[j,0]+EPSILON))
            p += tmp
            if not silent:
                print(j, x, tmp,  f"pi: {pis[j,0]}, mu: {mus[j,0]}, p: {tmp}")


        return p
    return mixture_pdf

def plot_density_overlay(pis, mus, sigmas, minimum, maximum, silent=True):
    mix_pdf = make_mixture_pdf(pis, mus, sigmas)
    bins = 20
    xs = np.linspace(minimum, maximum, 100)
    # ys = np.array([mix_pdf(xs[i], silent=silent and (i%100==0)) for i in range(xs.shape[0])])
    ys = np.array([mix_pdf(xs[i], silent=True) for i in range(xs.shape[0])])
    # print([(x, y) for x,y in zip(xs, ys)])
    plt.plot(xs,ys)#, label=f"iter={it}")

fig =  plt.figure()
plt.hist(data[0], bins=20,  density=True)
print("min", data.min(), "max", data.max())
print("stat_dist_hat:", stat_dist_hat.T)
print("mus_hat:", mus_hat)
print("mus_hat var:", mus_samples.var(axis=1))
print("sigma2s_hat:", sigma2s_hat)
plot_density_overlay(stat_dist_hat.T, mus_hat, np.sqrt(sigma2s_hat), data.min(), data.max())
fig.savefig(f"data_hist_k{K}.png")






