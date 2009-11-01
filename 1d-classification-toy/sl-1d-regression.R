#
# R script to duplicate a simplified form of the linear regresion
# classifier in chapter 2 of hastie et.al.
#

# Generate traning data for two normally distributed classes.

# Input :
# size : size of the data in both sets

# l.mu, l.sd : mean and standard deviation for the normal distribution for class 1
#              class 1 maps to -1

# r.mu, r.sd : mean and standard deviation for the normal distribution for class 2
#              class 2 maps to 1

# output : LSM generated classification boundary based on the traing data.
#  	   Probability that data will be misclassified, based on an a priori
#	   estinmate of the classification boundary.
#          Graph of training data and LSM

	   
train = function(size, l.mu, l.sd, r.mu, r.sd) 
{
	pop.l.mu = l.mu
	pop.l.sd = l.sd

	
	pop.r.mu = r.mu
	pop.r.sd = r.sd

	S.est    = 0.5 * (r.mu + l.mu)

	prob.l =  1 - pnorm(S.est, pop.l.mu, pop.l.sd)
	prob.r =  pnorm(S.est, pop.r.mu, pop.r.sd)

	training.size = size

	data.l  = rnorm(training.size, pop.l.mu, pop.l.sd)
	data.r  = rnorm(training.size, pop.r.mu, pop.r.sd)

	data    = c(data.l, data.r)

	class.l = rep(-1, training.size)
	class.r = rep(1,  training.size)

	class   = c(class.l, class.r) 

	lm.r  = lm(class ~ data)
	g     = coef(lm.r)
	class = - g[1]/g[2]

	x.axis  = rep(0 , training.size)



	plot(data.l, x.axis, xlim = c(0,1), ylim = c(-1,1), col = 'red', pch = 25, 
		     xlab="data", ylab="", lab=c(10,5,5))
	points(data.l, class.l, xlim = c(0,1), ylim = c(-1,1), col = 'red', pch = 25)
	       
	points(data.r, x.axis, xlim = c(0,1), ylim = c(-1,1), col = 'blue', pch = 7)
        points(data.r, class.r, xlim = c(0,1), ylim = c(-1,1), col = 'blue', pch = 7)

        abline(lm.r, col = 'light green', lwd = 2)

	cat("coefficients : ", format(g), "\n" )
	cat("classifier : ", format(class, digits=4), "\n")
	cat("prob of misclassification of class 1 : ", format(prob.l, digits=2), "\n")
	cat("prob of misclassification of class 2 : ", format(prob.r, digits=2), "\n")

}

train.plot = function(filename, size, l.mu, l.sd, r.mu, r.sd) 
{
	png(file=filename, bg='white')	
	train(size, l.mu, l.sd, r.mu, r.sd)
	dev.off()
}

# Generate multiple estimates of the classification boundary for two normally 

# Input : 

# n :  number of simulations

# l.mu, l.sd : mean and standard deviation for the normal distribution for class 1
#              class 1 maps to -1

# r.mu, r.sd : mean and standard deviation for the normal distribution for class 2
#              class 2 maps to 1

# Output : vector of size n of classification boundary values.


sim = function(n, l.mu, l.sd, r.mu, r.sd)
{
	pop.l.mu = l.mu
	pop.l.sd = l.sd

	
	pop.r.mu = r.mu
	pop.r.sd = r.sd

	ans = c()

	for ( i in 1:n)
	{
		data.l  = rnorm(training.size, pop.l.mu, pop.l.sd)
		data.r  = rnorm(training.size, pop.r.mu, pop.r.sd)
		data    = c(data.l, data.r)

		class.l = rep(-1, training.size)
		class.r = rep(1,  training.size)
		class   = c(class.l, class.r) 

		lm.r = lm(class ~ data)
		g = coef(lm.r)
		class = - g[1]/g[2]
		ans = c(class, ans)
	}
	ans
}


