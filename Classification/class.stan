data {
  int<lower=1> N;                       
  int<lower=1> M;                     
  vector<lower=0, upper=1>[N] Y;     
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
  for (i in 1:M) {
    x[i] ~ normal(0, 1);
  }
  // Likelihood
  for (n in 1:N) {
    if(Y[n]>0.5){
      target += bernoulli_logit_lpmf(1 | logits[n]);
    } else {
      target += bernoulli_logit_lpmf(0 | logits[n]);
    }
  }
}







