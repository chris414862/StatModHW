# Gibbs iterations
m = 1000
burnin = 20
thinning_stride = 4
library("mvtnorm")
library("basicMCMCplots")

## (X_1, X_2) ~ MVN(mu=(0,2)^T, Sigma = ((1, .75)^T,(.75, 1)^T))
## (X_1|X_2=x_2)  ~ N(mu=0 + .75(x_2 - 2), sigma^2=(1-.75^2))
## (X_2|X_1=x_1)  ~ N(mu=2 + .75(x_1 - 0), sigma^2=(1-.75^2))

mu1 = 0
mu2 = 2
sigma_21 = 1
sigma_22 = 1
cov=.75
rho=cov/((sigma_21*sigma_22)**(1/2))
sigma_21_cond = (1-rho**2)*sigma_21
sigma_22_cond = (1-rho**2)*sigma_22
print(paste("sigma_cond:", sigma_21_cond))

x1_cond_sampler = function(x2=NULL){
    return(rnorm(n=1, mean=mu1+rho*(x2-mu2) , sd=(sigma_21_cond**(1/2))))
}
x2_cond_sampler = function(x1=NULL){
    return(rnorm(n=1, mean=mu2+rho*(x1-mu1) , sd=(sigma_22_cond**(1/2))))#.661438))
}
prev_x1 = 0
prev_x2 = 0

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
grid_num <- 100
x = seq(from=-3, to=3, length.out=grid_num)
y = seq(from=-1, to=5, length.out=grid_num)
Sigma = rbind(c(1  , .75),
              c(.75, 1  ))


z = expand.grid(wt=x, hp=y)
z = dmvnorm(z, mean=c(0, 2), sigma=Sigma)
z = data.matrix(z)
z = matrix(z, grid_num, grid_num, byrow=T)

# print("samps")
# print(str(samps))
png("plots/gibbs_samples.png")
plot(x1_samps, x2_samps)
contour(x,y,z, add=TRUE)

png("plots/x1_trace_plot.png")
plot(x1_samps, type="l")
png("plots/x2_trace_plot.png")
plot(x2_samps, type="l")
png("plots/x1_lag1_autocorr.png")
acf(x1_samps, lag.max=20)
