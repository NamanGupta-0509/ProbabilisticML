data {
  int<lower=1> N;                       
  int<lower=1> M;                     
  array[N] int<lower=0, upper=1> Y;     
  matrix[N, M] H;               
}
parameters {
  vector[M] x;                   
}
transformed parameters {
  vector[N] logits = H * x;             // Linear predictor
}            
model {
  // Prior for x
  x ~ normal(0,1);

  // Likelihood
  Y ~ bernoulli_logit(logits);
}







