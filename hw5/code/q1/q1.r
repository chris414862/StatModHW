m = 10000
n = 4

deg_free = n-1
prop_dist = dnorm
prop_dist_sampler = rnorm
target_dist = function(m){
    return(dt(m, df=deg_free))
}

# Using importance resampling draw n samples from p(theta|data) = t_3(theta) 

get_resamples = function(m, prop_dist, prop_dist_sampler, target_dist){
    # first draw m samples from proposal dist
    prop_samps <- prop_dist_sampler(m)
    dim(prop_samps) <- c(m,1)

    # next compute weights
    weights <- target_dist(prop_samps)/ prop_dist(prop_samps) 

    # normalize weights
    sum = colSums(weights)
    norm_weights = weights/sum

    # re-sample originals from the dist defined by the normalized weights
    resamples = sample(prop_samps, m, prob=norm_weights, replace=TRUE)
    dim(resamples) <- c(m,1)
    return(data.frame(resamples, prop_samps, weights, norm_weights))
}

plot_true_dist_hist = function(samples, true_dist, m=100){

    # Make histogram
    hist(samples, prob=TRUE, xlim=c(-4, 4), col="green", main=paste("Samples:",toString(m), sep=" "))

    # hist(samples, prob=TRUE, xlim=c(-4, 4), col="green", main="Samples: 100")
    curve(true_dist(x), from=-4, to=4, add=TRUE, col="red")
    legend(-4, .3, legend=c("resamples", "true dist"), fill=c("green", "red"))
}
png("plots/resample_m100.png")
resamples = get_resamples(100, prop_dist, prop_dist_sampler, target_dist)
plot_true_dist_hist(resamples$resamples, target_dist, m=100)

png("plots/resample_m10000.png")
resamples = get_resamples(10000, prop_dist, prop_dist_sampler, target_dist)
plot_true_dist_hist(resamples$resamples, target_dist, m=10000)

# estimate E[theta|data]
m = 100
rs = get_resamples(m, prop_dist, prop_dist_sampler, target_dist)
mu_hat = (rs$prop_samps %*% rs$norm_weights)/m
mu_hat2 = mean(rs$resamples)
print("mu: 0")
print(paste("m=100: mu_hat", toString(mu_hat)))
print(paste("m=100: mu_hat2", toString(mu_hat2)))

m = 10000
rs = get_resamples(m, prop_dist, prop_dist_sampler, target_dist)
mu_hat = (rs$prop_samps %*% rs$norm_weights)/m
mu_hat2 = mean(rs$resamples)
print(paste("m=10000: mu_hat", toString(mu_hat)))
print(paste("m=10000: mu_hat2", toString(mu_hat2)))
