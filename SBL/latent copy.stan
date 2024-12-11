data {
  int<lower=1> N;             
  int<lower=1> M;             
  vector[N] Y_real;      
  vector[N] Y_imag;           
  matrix[N, M] H_real;        
  matrix[N, M] H_imag;        
  real<lower=0> sigma;   
  real<lower=1e-6> alpha;
}

parameters {
  vector[M] x_real;           
  vector[M] x_imag;           
  // vector<lower=1e-6>[M] alpha;
}

transformed parameters {
  vector[M] alpha_inv = sqrt(1/alpha);
}

model {
  // Prior for x
  for (i in 1:M) {
    x_real[i] ~ normal(0, alpha[i]);
    x_imag[i] ~ normal(0, alpha[i]);
  }

  alpha ~ gamma(1e-6, 1e-6);

  // Likelihood for Y
  Y_real ~ normal(H_real * x_real - H_imag * x_imag, sigma);
  Y_imag ~ normal(H_real * x_imag + H_imag * x_real, sigma);
}




