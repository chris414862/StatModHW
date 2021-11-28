# Gibbs iterations
m = 30
burnin = 10
thinning_stride = 2

## (X_1, X_2) ~ MVN(mu=(0,2)^T, Sigma = ((1, .75)^T,(.75, 1)^T))
## (X_1|X_2=x_2)  ~ N(mu=0 - .75(x_2 - 2), sigma^2=.4375)
## (X_2|X_1=x_1)  ~ N(mu=2 - .75(x_1 - 0), sigma^2=.4375)

mu1 = 0
mu2 = 2
sigma_21 = 1
sigma_22 = 1
cov=.75
sigma_2_cond = sigma_21*sigma_22 - (cov*cov)
sigma_cond = sigma_2_cond**(1/2)
print(paste("sigma_cond:", toString(sigma_cond)))

x1_cond_sampler = function(x2=NULL){
    return(rnorm(n=1, mean=mu1-.75*(x2-mu2) , sd=sigma_cond))
}
x2_cond_sampler = function(x1=NULL){
    return(rnorm(n=1, mean=mu2-.75*(x1-mu1) , sd=sigma_cond))#.661438))
}
prev_x1 = mu1
prev_x2 = mu2

# x1_samp = x1_cond_sampler(x2=x2_prev)
# x2_samp = x2_cond_sampler(x1=x1_prev)
# print(paste("x2_samp:", toString(x2_samp)))


# x1_samps = vector(mode="list")
# x2_samps = vector(mode="list")
x1_samps = vector()#(mode="list")
x2_samps = vector()#(mode="list")


for (i in 1:m){
    curr_x1 <- x1_cond_sampler(x2=prev_x2)
    x1_samps <- append(x1_samps, curr_x1[1])
    curr_x2 <- x2_cond_sampler(x1=curr_x1)
    x2_samps <- append(x2_samps, curr_x2)
    prev_x1 <- curr_x1
    prev_x2 <- curr_x2
}

# remove burnin
x1_samps <- x1_samps[seq(from=burnin+1, to=length(x1_samps))]
x2_samps <- x2_samps[seq(from=burnin+1, to=length(x2_samps))]

# use thinning
x1_samps <- x1_samps[seq(from=1, to=length(x1_samps), by=thinning_stride)]
x2_samps <- x2_samps[seq(from=1, to=length(x2_samps), by=thinning_stride)]

samps <- rbind(x1_samps, x2_samps)

# TODO: Make contour plot

# print("samps")
print(str(samps))
png("plots/gibbs_samples.png")
plot(x1_samps, x2_samps)


