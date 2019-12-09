saol <- function(data, modelmat, real = FALSE) {
  max_k         <- ceiling(log2(n))
  max_k_experts <- ceiling(n / 2^seq(0, max_k))
  weights       <- rep(pmin(1 / sqrt(2^seq(0, max_k) - 1), .5), max_k_experts)
  ada_weights <- matrix(1, ncol = sum(max_k_experts), nrow = 2)

  ada_loss      <- numeric(n)

  for (j in seq(nrow(data))) {
    active <- NULL
    active_lengths <- NULL
    for (i in 0:max_k) {
      k <- max_k_experts[i + 1]
      eta   <- min(1 / sqrt(2^i - 1), .5)
      start <- seq(k) * 2^i
      end   <- (seq(k)+1) * 2^i
      idx <- which(start <= j & end >= j)
      active <- c(active, idx + sum(max_k_experts[seq(0, i)]))
      active_lengths <- c(active_lengths, end[idx] - start[idx] + 1)
    }

    preds <- drop(
      t(ada_weights[, active, drop = FALSE]) %*% modelmat[j, ])
  
    ada_pred <- sample(preds, 1, prob = weights[active] / sum(weights[active]))
    ada_loss[j]         <- (ada_pred - data$y[j])^2

    ada_grads   <- -2 * cbind(data$y[j] - preds) %*% t(modelmat[j, ])
    new_weights <- t(ada_weights[, active]) - pmin(
      .1 / sqrt(active_lengths), .5) * ada_grads
    ada_weights[1, active] <- new_weights[, 1]
    ada_weights[2, active] <- new_weights[, 2]

    expert_loss         <- (preds - data$y[j])^2

    if (isTRUE(real)) {
      instant_regret  <- (ada_pred - data$y[j])^2 - preds
      weights[active] <- weights[active] * (1 + eta * instant_regret)
    } else {
      weights[active] <- weights[active] * (1 - eta)^expert_loss
    }
    weights[active]    <- replace_na(weights[active], 0)
  }
  ada_loss
}
