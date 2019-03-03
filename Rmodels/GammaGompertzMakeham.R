##' The Gamma-Gompertz-Makeham distribution
##' 
##' Density, distribution function, hazards, quantile function and random
##' generation for the Gamma-Gompertz-Makeham distribution.
##' 
##' The Gamma-Gompertz-Makeham distribution with shape parameters \code{lambda}
##' \code{s} and \code{beta} \eqn{\beta, s, \lambda > 0}{beta,s,lambda > 0} 
##' and \rate{rate} parameter \eqn{b>0} has probability density function
##' 
##' \deqn{f(x | b,\beta,s,\lambda) = \beta^s e^{-(\lambda+bs)x} \frac{ bs + 
##'	\lambda \lbrack 1+(\beta-1)e^{-bx} \rbrack }{ 
##' {\lbrack 1+(\beta-1)e^{-bx} \rbrack}^s} }
##' {f(x | b,beta,s,lambda) = beta^s*exp[-(lambda+bs)x]*
##'	[bs+lambda+lambda(beta-1)e^(-bx)]/[1+(beta-1)e^(-bx)]^s }
##' 
##' and hazard
##' 
##' \deqn{ h(x | b,\beta,s,\lambda) = \lambda+\frac{bs}{1+(\beta-1)e^{-bx}} }{
##'	h(x | b,beta,s,lambda) = lambda+(bs)/(1+(beta-1)exp(-bx)) }
##' 
##' The hazard is increasing for shape \eqn{\beta>1}{beta>1} and decreasing for
##' \eqn{0<\beta<1}{0<beta<1}. For \eqn{\beta=1}{beta=1}, the 
##' Gamma-Gompertz-Makeham is equivalent to the exponential distribution with 
##' constant hazard rate  \eqn{bs+\lambda}.
##' 
##' The probability distribution function is 
##'
##' \deqn{ F(x|b,\beta,s,\lambda) = 1-e^{-\lambda x}\lbrack 1+\frac{1}{\beta}
##' (e^{-bx}-1) \rbrack^{-s} }{ F(x| b,beta,s,lambda) = 1-exp(-lambda*x)*
##'	[1+(exp(bx)-1)/beta]^(-s)  }
##' 
##' @aliases GammaGompertzMakeham dgammagompertzmakeham pgammagompertzmakeham 
##' qgammagompertzmakeham hgammagompertzmakeham Hgammagompertzmakeham
##' rgammagompertz
##' @param x,q vector of quantiles.
##' @param p vector of probabilities.
##' @param n number of observations. If \code{length(n) > 1}, the length is
##' taken to be the number required.
##' @param rate Vector of rate parameters, reciprocal of scale.
##' @param beta Vector of shape parameter beta.
##' @param s Vector of shape parameter s.
##' @param log,log.p logical; if TRUE, probabilities p are given as log(p).
##' @param lower.tail logical; if TRUE (default), probabilities are \eqn{P(X
##' \le x)}{P(X <= x)}, otherwise, \eqn{P(X > x)}{P(X > x)}.
##' @return 
##' \code{dgammagompertzmakeham} gives the density, 
##' \code{pgammagompertzmakeham} gives the distribution function, 
##' \code{hgammagompertzmakeham} gives the hazard function,
##' \code{Hgammagompertzmakeham} gives the cumulative hazard function, and 
##' \code{rgammagompertzmakeham} generates random deviates.
##' 
##' @author Yifan F. Yang <yifan.yang@inserm.fr>
##' @seealso \code{\link{dgammagompertz}}
##' @references 
##' @keywords distribution
##' @name GammaGompertzMakeham
NULL


##' @export
##' @rdname GammaGompertzMakeham
dgammagompertzmakeham <- function(x, rate=1, beta, s, lambda, log.p=FALSE) {

    if ( any(c(rate, beta, s) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }

    y <- ifelse(x <= 0, 0, x * rate)
	z <- ifelse(x <= 0, 0, expm1(y)/beta )
	l <- (rate*s+lambda)*exp(y) + lambda*(beta-1)

    if (log.p){
        ret <- ifelse(x < 0,
                      0,
					  log(l) - log(beta) - lambda*x - (s+1)*log1p(z)  )
					  # for GG: log(rate) + log(s) - log(beta) + y - (s+1)*log1p(z) )
    }else{
        ret <- ifelse(x < 0,
                      0,
					  l * exp((-s-1)*log1p(z)) * exp(-lambda*x) / beta)
					  # for GG: rate * s * exp(y) * exp((-s-1)*log1p(z)) / beta )
    }

    return ( ret )
}

##' @export
##' @rdname GammaGompertzMakeham
pgammagompertzmakeham <- function(q, rate=1, beta, s, lambda, lower.tail = TRUE, log.p = FALSE) {
    
	n <- length(q)
	
    if ( any(c(rate, beta, s) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }

    y <- ifelse(q <= 0, 0, q * rate)
	z <- ifelse(q <= 0, 0, expm1(y)/beta )
	H <- ifelse(q <= 0, 0, lambda*q+s*log1p(z) )

    if (log.p){
        if (lower.tail){
            ret <- ifelse(q <= 0,
                          -Inf,
						  log(-expm1(-H))
                          )
        }else{
            ret <- ifelse(q <= 0,
                          0,
                         -H
                          )
        }
    }else{
        if (lower.tail){
            ret <- ifelse(q <= 0,
                          0,
                          -expm1(-H)  #1-S(x) = 1-exp(-H(x))
                          )
        }else{
            ret <- ifelse(q <= 0,
                          1,
                          exp(-H) #  S(x)=exp(-H(x))
                          )
        }
    }

    return ( ret )
}

##' @export
##' @rdname GammaGompertzMakeham
qgammagompertzmakeham <- function(p, rate=1, beta, s, lambda, lower.tail = TRUE, log.p = FALSE) {
	
    if ( any(c(rate, beta, s) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }
	
    warning("Place holder quantile function, will always return 0 or NaN")
	
    if (log.p) p <- exp(p)

    ok <- (p >= 0) & (p <= 1)

    ret <- ifelse(ok,
				#No direct formula to caculate q, have to numerical solve an equation
				#For GG: log1p(beta*expm1(-log1p(-p)/s))/rate,
				0,
				NaN)

    if (!all(ok)) warning("qgammagompertzmakeham produced NaN's")

    return ( ret )
}

##' @export
##' @rdname GammaGompertzMakeham
rgammagompertzmakeham <- function(n, rate=1, beta, s, lambda){

    if ( any(c(rate, beta, s) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }

    
    y <- runif(n)

    return ( qgammagompertzmakeham(y, rate, beta, s, lambda) )
}

##' @export
##' @rdname GammaGompertzMakeham
hgammagompertzmakeham <- function(x, rate=1, beta, s, lambda, log.p = FALSE) 
{
	
    if ( any(c(rate, beta, s) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }
    
    y <- ifelse(x <= 0, 0, x * rate)
	u <- ifelse(x <= 0 | beta==1, 0 , (beta-1)*exp(-y))
	
	ret <- ifelse(x<0, 0, lambda+(rate*s)/(1+u) )
  
	if (log.p) ret <- log(ret)
		
    return ( ret )
}

##' @export
##' @rdname GammaGompertzMakeham
Hgammagompertzmakeham <- function(x, rate=1, beta, s, lambda, log.p = FALSE) 
{
    if ( any(c(rate, beta, s) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }
    
    y <- ifelse(x <= 0, 0, x * rate)
	z <- ifelse(x <= 0, 0, expm1(y)/beta )
	ret <- ifelse(x <= 0, 0, lambda*x+s*log1p(z) )

    if (log.p) ret <- log(ret)

    return ( ret )
}

##' @export
##' @rdname means
rmst_gammagompertzmakeham = function(t, rate=1, beta, s, lambda, start=0){
  rmst_generic(pgammagompertzmakeham, t, start=start, beta=beta, s=s, rate=rate)
}

##' @export
##' @rdname means
mean_gammagompertzmakeham = function(rate=1, beta, s, lambda){
  rmst_generic(pgammagompertzmakeham, Inf, start=0, beta=beta, s=s, rate=rate)
}

gammagompertzmakeham = list(
             name="gammagompertzmakeham",
             pars=c("rate","beta","s","lambda"),
             location="rate",
             transforms=c(log, log, log, log),
             inv.transforms=c(exp, exp, exp, exp),
             inits=function(t){
				 c(1 / mean(t), 100, 1, 0.001)
             }
        )