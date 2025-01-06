data {
  int<lower=1> N;             
  int<lower=1> M;             
  vector[N] Y_real;      
  vector[N] Y_imag;           
  matrix[N, M] H_real;        
  matrix[N, M] H_imag;        
  real<lower=0> sigma;
        
}

parameters {
  vector[M] x_real;           
  vector[M] x_imag;
  vector<lower=1e-6>[M] alpha;               
}

transformed parameters {
  vector[M] alpha_inv = sqrt(1/alpha);
}

model {
  // Priors for x_real and x_imag
  x_real ~ normal(0, alpha_inv);
  x_imag ~ normal(0, alpha_inv);

  alpha ~ gamma(1e-6, 1e-6);

  // Likelihood for the real and imaginary parts of Y
  Y_real ~ normal(H_real * x_real - H_imag * x_imag, sigma);
  Y_imag ~ normal(H_real * x_imag + H_imag * x_real, sigma);
}


