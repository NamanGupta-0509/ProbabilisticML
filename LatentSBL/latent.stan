data {
  int<lower=1> N;         
  int<lower=1> M;         
  vector[N] Y;            
  matrix[N, M] H;        
  real<lower=0> sigma;
}

parameters {
  vector[M] x;           
  vector<lower=1e-6>[M] alpha;
}

transformed parameters {
  vector[M] alpha_inv = sqrt(1/alpha);
}


model {
  // Prior for x
  // for (i in 1:M) {
  //   x[i] ~ normal(0, alpha_inv[i]);
  // } 

  x ~ normal(0, alpha_inv);

  // target += -log(alpha);
  
  alpha ~ gamma(1e-6, 1e-6);



  // Likelihood for Y
  Y ~ normal(H * x, sigma);
}





