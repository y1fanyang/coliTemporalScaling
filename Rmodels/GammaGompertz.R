##' The Gamma-Gompertz distribution
##' 
##' Density, distribution function, hazards, quantile function and random
##' generation for the Gamma-Gompertz distribution with unrestricted shape.
##' 
##' The Gamma-Gompertz distribution with shape parameters \code{beta} and
##' \code{s}, \eqn{s, \beta > 0}{s, beta > 0} and \rate{rate} parameter 
##' \eqn{b>0} has probability density function
##' 
##' \deqn{f(x | b,s,\beta) = \frac{bse^{bs}\beta^s}{{(\beta-1+e^{bx})}^{s+1}} }   
##' {f(x | b,s,beta) = [bs exp(bx) beta^s]/[beta-1+exp(bx)]^(s+1) }
##' 
##' and hazard
##' 
##' \deqn{h(x | b, s, \beta) = \frac{bs}{1+(\beta-1)e^{-bx}} }{ h(x | b, 
##'	s, beta) = (bs)/(1+(beta-1)exp(-bx))}
##' 
##' The hazard is increasing for shape \eqn{\beta>1}{beta>1} and decreasing for
##' \eqn{0<\beta<1}{0<beta<1}. For \eqn{\beta=1}{beta=1}, the Gamma-Gompertz is
##' equivalent to the exponential distribution with constant hazard rate 
##' \eqn{bs}.
##' 
##' The probability distribution function is 
##'
##' \deqn{ F(x | b, s, \beta) = 1 - \frac{\beta^s}{{(\beta-1+e^{bx})}^s} }
##' {F(x | b, s, beta) = 1 - beta^s/[beta-1+exp(bx)]^s }
##' 
##' @aliases GammaGompertz dgammagompertz pgammagompertz qgammagompertz 
##' hgammagompertz Hgammagompertz rgammagompertz
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
##' \code{dgammagompertz} gives the density, 
##' \code{pgammagompertz} gives the distribution function, 
##' \code{qgammagompertz} gives the quantile function, 
##' \code{hgammagompertz} gives the hazard function,
##' \code{Hgammagompertz} gives the cumulative hazard function, and 
##' \code{rgammagompertz} generates random deviates.
##' 
##' @author Yifan F. Yang <yifan.yang@inserm.fr>
##' @seealso \code{\link{dexp}} \code{\link{dgompertz}}
##' @references Manton, K. G.; Stallard, E.; Vaupel, J. W. (1986) Alternative
##' Models for the Heterogeneity of Mortality Risks Among the Aged. Journal of
##' the American Statistical Association. 81: 635–644. 
##' doi:10.1080/01621459.1986.10478316
##' @keywords distribution
##' @name GammaGompertz
NULL


##' @export
##' @rdname GammaGompertz
dgammagompertz <- function(x, beta, s, rate=1, log.p=FALSE) {

    if ( any(c(beta, s, rate) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }

    y <- ifelse(x <= 0, 0, x * rate)
	z <- ifelse(x <= 0, 0, expm1(y)/beta )

    if (log.p){
        ret <- ifelse(x < 0,
                      0,
					  log(rate) + log(s) - log(beta) + y - (s+1)*log1p(z) )
    }else{
        ret <- ifelse(x < 0,
                      0,
					  rate * s * exp(y) * exp((-s-1)*log1p(z)) / beta )
    }

    return ( ret )
}

##' @export
##' @rdname GammaGompertz
pgammagompertz <- function(q, beta, s, rate=1, lower.tail = TRUE, log.p = FALSE) {
    n <- length(q)
	
    if ( any(c(beta, s, rate) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }

    y <- ifelse(q <= 0, 0, q * rate)
	z <- ifelse(q <= 0, 0, expm1(y)/beta )
	H <- ifelse(q <= 0, 0, s*log1p(z) )

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
                          -expm1(-H) # = 1-1/(1+z)^s
                          )
        }else{
            ret <- ifelse(q <= 0,
                          1,
                          exp(-H) #  = 1/(1+z)^s
                          )
        }
    }

    return ( ret )
}

##' @export
##' @rdname GammaGompertz
qgammagompertz <- function(p, beta, s, rate=1, lower.tail = TRUE, log.p = FALSE) {

    if ( any(c(beta, s, rate) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }
    
    if (log.p) p <- exp(p)

    ok <- (p >= 0) & (p <= 1)


    ret <- ifelse(ok,
				log1p(beta*expm1(-log1p(-p)/s))/rate,
				NaN)

    if (!all(ok)) warning("qgammagompertz produced NaN's")

    return ( ret )
}

##' @export
##' @rdname GammaGompertz
rgammagompertz <- function(n, beta, s = 1, rate = 1){
    if ( any(c(beta, s, rate) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }

    
    y <- runif(n)

    return ( qgammagompertz(y, beta, s, rate) )
}

##' @export
##' @rdname GammaGompertz
hgammagompertz <- function(x, beta, s, rate = 1, log.p = FALSE) 
{
	
    if ( any(c(beta, s, rate) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }
    
    y <- ifelse(x <= 0, 0, x * rate)
	u <- ifelse(x <= 0 | beta==1, 0 , (beta-1)*exp(-y))
	
    if (log.p) {
       		ret <- ifelse(x < 0,
                      -Inf,
                      log(rate)+log(s)-log1p(u)
                      )
    }else{
        	ret <- ifelse(x < 0,
                      0,
                      (rate*s)/(1+u)
                      )
    }
	
    return ( ret )
}

##' @export
##' @rdname GammaGompertz
Hgammagompertz <- function(x, beta, s, rate = 1, log.p = FALSE) 
{
    if ( any(c(beta, s, rate) <= 0) ){
        warning("Non-positive shape or rate")
        return(NaN)
    }
    
    y <- ifelse(x <= 0, 0, x * rate)
	z <- ifelse(x <= 0, 0, expm1(y)/beta )
	ret <- ifelse(x <= 0, 0, s*log1p(z) )

    if (log.p) ret <- log(ret)

    return ( ret )
}

##' @export
##' @rdname means
rmst_gammagompertz = function(t, beta, s, rate=1, start=0){
  rmst_generic(pgammagompertz, t, start=start, beta=beta, s=s, rate=rate)
}

##' @export
##' @rdname means
mean_gammagompertz = function(beta, s, rate = 1){
  rmst_generic(pgammagompertz, Inf, start=0, beta=beta, s=s, rate=rate)
}

gammagompertz = list(
             name="gammagompertz",
             pars=c("beta","s","rate"),
             location="rate",
             transforms=c(log, log, log),
             inv.transforms=c(exp, exp, exp),
             inits=function(t){
				 c(100, 0.1, 1 / mean(t))
             }
        )