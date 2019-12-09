ada_new <- function(data, modelmat) {
  ada_weights <- matrix(1, ncol = n, nrow = 2)
  ada_probs   <- rep(1, n) #numeric(n)
  ada_loss    <- numeric(n)
  # eta         <- 1/100
  cum_losses  <- numeric(n)
  
  eta <- 1 / log(nrow(data))
  
  for (i in seq(n)) {
    preds       <- drop(
      t(ada_weights[, seq(i), drop = FALSE]) %*% modelmat[i, ])
    ada_pred    <- sum((ada_probs[seq(i)] / sum(ada_probs[seq(i)])) * preds)
    ada_pred <- sample(preds, 1, prob = ada_probs[seq(i)] / sum(ada_probs[seq(i)])) 
    # ada_pred <- sample(preds, 1, prob = ada_probs[seq(i)] / sum(ada_probs[seq(i)]))
    ada_loss[i] <- (ada_pred - data$y[i])^2
    
    # Update individual experts
    ada_grads   <- -2 * cbind(data$y[i] - preds) %*% t(modelmat[i, ])
    new_weights <- t(ada_weights[, seq(i)]) - pmin(
      .1 / sqrt(rev(seq(i))), .5) * ada_grads
    ada_weights[1, seq(i)] <- new_weights[, 1]
    ada_weights[2, seq(i)] <- new_weights[, 2]
    
    # Update expert tracking
    expert_loss         <- (preds - data$y[i])^2
    ada_probs[seq(i)]   <- ada_probs[seq(i)] * (1+eta)^(ada_loss[i] - expert_loss)#(1 - eta)^expert_loss
    
    ada_probs[seq(i)] <- replace_na(ada_probs[seq(i)], 0)
  }
  ada_loss
}
