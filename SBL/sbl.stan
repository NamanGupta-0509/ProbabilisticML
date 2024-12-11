data {
  int<lower=1> N;         // Number of observations (rows of H)
  int<lower=1> M;         // Number of features (columns of H)
  vector[N] Y;            // Single observed data vector y
  matrix[N, M] H;         // Fixed measurement matrix
  real<lower=0> sigma;    // Noise standard deviation
  vector[M] alpha;
}

parameters {
  vector[M] x;            // Sparse vector to infer
}

model {
  // Prior for x
  for (i in 1:M) {
    x[i] ~ normal(0, alpha[i]);
  }

  // Likelihood for Y: Gaussian noise with standard deviation sigma
  Y ~ normal(H * x, sigma);
}


