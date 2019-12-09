library(MASS)
library(tidyverse)
library(magrittr)
library(cowplot)
theme_set(theme_cowplot())
set.seed(99)

# Set up data
Sigma      <- matrix(c(1,.9,.9, 1),2,2)
base_data  <- mvrnorm(n = 1500, rep(0, 2), Sigma) %>%
  set_colnames(c("x", "y")) %>% 
  as_tibble()

n     <- nrow(base_data)

# Corrupt data
corrupt_data <- function(data, gen_xy, corr_idx_existing) {
  corr_idx   <- sample(setdiff(seq(n), corr_idx_existing), 1)
  data[corr_idx, ] <- gen_xy()
  list(data, c(corr_idx_existing, corr_idx))
}

# halve_data <- function(data, gen_xy, corr_idx_existing) {
#   if (is.null(corr_idx_existing)) {
#     corr_idx <- as.integer(n/2)
#   } else {
#     corr_idx <- which.max(c(corr_idx_existing, n) - c(0, corr_idx_existing))[1]
#     corr_idx <- ceiling((c(0,corr_idx_existing, n)[corr_idx+1] - c(0, corr_idx_existing, n)[corr_idx])/2)
#   }
#   data[corr_idx, ] <- gen_xy()
#   list(data, sort(c(corr_idx_existing, corr_idx)))
# }

standard_gd <- function(b, data, modelmat) {
  gd_loss <- numeric(n)
  for (i in seq(n)) {
    
    gd_loss[i] <- (modelmat[i, ] %*% t(b) - data$y[i])^2
    mm <- modelmat[i, ]

    grad <- -2 * data$y[i] - mm %*% t(b) %*% mm
    eta  <- .1 / sqrt(i)
    b    <- b - eta * grad
  }
  gd_loss
}

# Adaptive GD
adaptive_gd <- function(data, modelmat, idea = FALSE) {
  ada_weights <- matrix(1, ncol = n, nrow = 2)
  ada_probs   <- numeric(n)
  ada_loss    <- numeric(n)
  # eta         <- 1/100
  cum_losses  <- numeric(n)
  
  for (i in seq(n)) {
    if (i == 1) {
      eta <- 0
    } else if (isTRUE(idea)) {
      eta <- 1 / sqrt(sum(rev(seq(i)) * ada_probs[seq(i)]))
      } else {
        eta <- 1/sqrt(i)
      }
    preds       <- drop(
      t(ada_weights[, seq(i), drop = FALSE]) %*% modelmat[i, ])
    # cum_losses[seq(i)] <- cum_losses[seq(i)] + (preds - data$y[i])^2 - sqrt(i) * rexp(i)
    # ada_pred    <- sum(ada_probs[seq(i)] * preds)
    if (i == 1) {
      ada_pred <- preds[1]
    } else {
      ada_pred <- sample(preds, 1, prob = ada_probs[seq(i)] / sum(ada_probs[seq(i)]))
    }
    
    ada_loss[i] <- (ada_pred - data$y[i])^2
    
    # Update individual experts
    
    ada_grads   <- -2 * cbind(data$y[i] - preds) %*% t(modelmat[i, ])
    new_weights <- t(ada_weights[, seq(i)]) - pmin(
      .1 / sqrt(rev(seq(i))), .5) * ada_grads
    ada_weights[1, seq(i)] <- new_weights[, 1]
    ada_weights[2, seq(i)] <- new_weights[, 2]
    
    # Update expert tracking
    
    expert_loss         <- (preds - data$y[i])^2
    ada_probs[seq(i)]   <- ada_probs[seq(i)] * exp(-eta * expert_loss)
    
    ada_probs[seq(i)] <- ada_probs[seq(i)] / sum(ada_probs[seq(i)])
    ada_probs[seq(i)] <- replace_na(ada_probs[seq(i)], 0)
    
    # Update weights to initialize new round
    ada_probs[seq(i)]   <- ada_probs[seq(i)] * (1 - 1 / (i + 1))
    ada_probs[i + 1]    <- 1 / (i + 1)
    ada_probs[seq(i)] <- replace_na(ada_probs[seq(i)], 0)
  }
  ada_loss
}

get_ada_intervals <- function(corr_idx) {
  ada_idx <- sort(corr_idx)
  ada_intervals <- which(ada_idx - c(ada_idx[-1], 1000) < -1)
  bind_cols("start" = ada_idx[ada_intervals],
            "end"   = c(ada_idx, 1000)[ada_intervals + 1])
}

get_ada_regret <- function(ada_intervals, loss, optimal_preds) {
  pmap_dbl(ada_intervals, ~ {
    optimal_model <- lm(y ~ x, base_data[seq(.x, .y), ])$fitted_values
    sum(loss[seq(.x, .y)]) - sum(optimal_model - base_data$y[seq(.x, .y)])
    }) %>%
    max()
}

epsilon <- 500
repetitions <- 3
exp_losses     <- matrix(0, ncol = epsilon, nrow = repetitions)
gd_losses      <- matrix(0, ncol = epsilon, nrow = repetitions)
ada_losses     <- matrix(0, ncol = epsilon, nrow = repetitions)

ada_losses_idea <- matrix(0, ncol = epsilon, nrow = repetitions)

saol_losses2 <- matrix(0, ncol = epsilon, nrow = repetitions)

for (j in 1:repetitions) {
  cat(j, "/", repetitions, "\n")


  corr_idx       <- NULL
  corrupted_data <- base_data
  
  for (i in seq(epsilon)) {
    # if (i %% 100 == 0) {
      cat(j, "/", repetitions, "--", i, "/", epsilon, "\n")
    # }
    corrupted_data <- corrupt_data( #corrupt_data(
      corrupted_data,
      function() mvrnorm(n = 1, rep(3, 2), Sigma), corr_idx)#c(rexp(1, .75), rt(1, 1, 5)), corr_idx)
    corr_idx <- corrupted_data[[2]]
    corrupted_data <- corrupted_data[[1]]
  
    modelmat <- cbind(1, corrupted_data$x)
    exp_losses[i] <- sum(((
      model.matrix(y~x, base_data[-corr_idx, ]) %*% coef(
      lm(y~x, base_data[-corr_idx, ])) - base_data$y[-corr_idx]))^2)
  
    gd_loss <- standard_gd(cbind(1, 1), corrupted_data, modelmat)
    ada_loss <- adaptive_gd(corrupted_data, modelmat)

    ada_loss_idea <- ada_new(corrupted_data, modelmat)
 
    saol_loss2 <- saol(corrupted_data, modelmat, real = TRUE)
    
    
    # SAOL_loss <- SAOL(corrupted_data, modelmat)
    
    gd_losses[j, i]       <- sum(gd_loss[-corr_idx])
    ada_losses[i]         <- sum(ada_loss[-corr_idx])
    ada_losses_idea[j, i] <- sum(ada_loss_idea[-corr_idx])
    
    # saol_losses[j, i]   <- sum(saol_loss[-corr_idx])
    saol_losses2[j, i]    <- sum(saol_loss2[-corr_idx])
   
    # SAOL_losses[i] <- sum(SAOL_loss[-corr_idx])
    # ada
    

  }
}

gd_losses1 <- colMeans(gd_losses)
saol_losses1 <- colMeans(saol_losses2)
ada_losses_idea1 <- colMeans(ada_losses_idea)

ada_losses_std1 <- colMeans(ada_losses)

full_regret_plt1 <- bind_cols(
  "Base Learner"  = gd_losses1,
  "SAOL (8)" = saol_losses1,
  # "Expert Tracking" = ada_losses_idea2,
  "AMW (6)" = ada_losses_idea1,
  "FLH (7)" = ada_losses_std1) %>%
  # mutate_all(~ .x - exp_losses) %>%
  mutate_all(~ .x / sqrt(seq(n - epsilon, 1))) %>%
  mutate(frac = seq(epsilon) / n) %>%
  gather(alg, regret, -frac) 

full_regret_plt2 <- bind_cols(
  # "Base Learner"  = gd_losses1,
  "SAOL (8)" = saol_losses1,
  # "Expert Tracking" = ada_losses_idea2,
  "AMW (6)" = ada_losses_idea1,
  "FLH (7)" = ada_losses_std1) %>%
  # mutate_all(~ .x - exp_losses) %>%
  mutate_all(~ .x / sqrt(seq(n - 2 * epsilon, 1))) %>%
  mutate(frac = seq(epsilon) / n) %>%
  gather(alg, regret, -frac) 

fpr1 <- ggplot(full_regret_plt1, aes(x = frac, y = regret, color = alg)) +
  geom_line(lwd=1) +
  labs(y = "Robust Regret", x = "Fraction of Outliers") +
  scale_colour_manual(
  values = c(
    "Base Learner" = "#8975ca", "SAOL (8)" =  "#71a659", "AMW (6)" = "#cb5683", "FLH (7)" = "#c5783e"), 
  name = "Algorithm") 

fpr2 <- ggplot(full_regret_plt2, aes(x = frac, y = regret, color = alg)) +
  geom_line(lwd=1) +
  labs(y = "Robust Regret", x = "Fraction of Outliers") +
  scale_colour_manual(
    values = c(
      "Base Learner" = "#8975ca", "SAOL (8)" =  "#71a659", "AMW (6)" = "#cb5683", "FLH (7)" = "#c5783e"), 
    name = "Algorithm") 
fpr2
fpr1 <- fpr1 + ylim(c(0, 1e10))
pgrid2 <- cowplot::plot_grid(fpr1 + theme(legend.position = "none") + xlab("") + scale_y_log10(), 
                             fpr2 + theme(legend.position = "none") + xlab("") + scale_y_log10(), nrow = 2, ncol = 1,
                   labels = "AUTO", align = "v")

legend_b <- get_legend(fpr1 + theme(legend.position="bottom"))
p <- plot_grid( pgrid2, legend_b, ncol = 1, rel_heights = c(1, .075))
p

cowplot::save_plot("ada_all_grid2.png", p, ncol = 3, nrow = 1, base_height = 8, base_width = 2.5)
cowplot::save_plot("full_regret2.png", 
                   fpr, base_aspect_ratio = 1.5,
                   base_height = 5)
empty_plt <- ggplot() + geom_blank()
legend_grid <- cowplot::plot_grid(
  empty_plt, plt_legend, empty_plt, rel_widths = c(.25, 3, .2))

plt_grid = cowplot::plot_grid(regret_plt + theme(
  legend.position = "none"), 
  full_regret_plt + theme(legend.position = "none"),
  align = "vh", labels = c("A", "B"), hjust = -1, nrow = 1)

pl_grid = cowplot::plot_grid(
  plt_grid, legend_grid, nrow = 2, rel_heights = c(3, .2))
pl_grid

cowplot::save_plot("ada_regret_duo.png", 
                   pl_grid, ncol = 2, nrow = 1)
