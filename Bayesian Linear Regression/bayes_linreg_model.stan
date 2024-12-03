data {
    int<lower=0> N;                     // the number of data points
    vector[N] x;                    
    vector[N] t;  
    matrix[N, N] K;    

    real<lower=0> a;
    real<lower=0> b;
    real<lower=0> c;
    real<lower=0> d;

}

parameters {
    vector<lower=0>[N] alpha;           // the i th alpha value to multiplied with K(xi,x)
    real<lower=0> alpha_0;              // the N+1 th alpha value -> to multiplied with 1

    real<lower=0> beta;   

    vector[N] w; 
    real<lower=0> w_0;  

}


model {
    // PRIORS
    alpha_0 ~ gamma(a,b);
    alpha ~ gamma(a, b);
    beta ~ gamma(c, d);

    // LIKELIHOOD
    w ~ normal(0, sqrt(1 ./ alpha));
    w_0 ~ normal(0, sqrt(1 ./ alpha));

    for(i in 1:N){
        t[i] ~ normal(K[i]*alpha + alpha_0, 1/beta);
    }
  
}


