import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import matplotlib


def plot_samples(num_samples_lst, plot_scale=2):
    plt.clf() # Clear previous figure

    # Create the subplots
    fig, axs_tup = plt.subplots(len(num_samples_lst),1)
    width, height = fig.get_size_inches()
    fig.set_size_inches(width*plot_scale,len(num_samples_lst)*height*plot_scale)

    # Sample and plot for each value in list
    for i, num_samples in enumerate(num_samples_lst):
        if len(num_samples_lst) > 1:
            ax = axs_tup[i]
        else:
            ax = axs_tup

        r = cauchy.rvs(size=num_samples)
        num_bins = int(r.max()-r.min())
        bin_counts, vals, _ = ax.hist(r, bins=num_bins, density=True)
        ax.set_title("Num Samples: "+str(num_samples_lst[i]))
        max_count = bin_counts.max()

        # Cutoff outliers to make the center visible
        if vals.min() < -100 or vals.max() > 100:
            ax.set_xlim(-50,50)

    return fig, ax


# Change font size
font = {'size'   : 18}
matplotlib.rc('font', **font)

# Experimental variables
# Adjust these however you wish. **More than 4 samplings might collapse the output 
num_samples_lst = [1000]
plot_scale = 1.5

# Make plots
fig, ax = plot_samples(num_samples_lst, plot_scale=plot_scale)
x = np.linspace(cauchy.ppf(0.01),
                cauchy.ppf(0.99), 100)

ax.plot(x, cauchy.pdf(x),
       'r-', lw=5, alpha=0.6)#;; label='cauchy pdf')
fig.savefig("cauchy_sampling_dist.png")


