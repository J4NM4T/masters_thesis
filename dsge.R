library(gEcon)

# ==============================================================================
# Wspolne parametry kalibracyjne
# ==============================================================================

beta_val   <- 0.99
sigma_val  <- 1.0
theta_val  <- 0.75
varphi_val <- 1.0
kappa_val  <- ((1 - theta_val) * (1 - theta_val * beta_val) / theta_val) *
              (sigma_val + varphi_val)

cat("\n================================================================\n")
cat("  Kalibracja:\n")
cat(sprintf("  beta = %.2f, sigma = %.1f, theta = %.2f, varphi = %.1f\n",
            beta_val, sigma_val, theta_val, varphi_val))
cat(sprintf("  kappa = %.6f\n", kappa_val))
cat("================================================================\n\n")


# ==============================================================================
# 1. STANDARDOWA REGULA TAYLORA
#    r[] = phi_pi * pi[] + phi_y * y[] + eps_R[]
# ==============================================================================

cat("########################################\n")
cat("# MODEL 1: Standardowa regula Taylora #\n")
cat("########################################\n\n")

model_code_1 <- paste0("
block EQUILIBRIUM {
    identities {
        y[]     = E[][y[1]] - (1/sigma)*(r[] - E[][pi[1]]) + eps_D[];
        pi[]    = beta*E[][pi[1]] + kappa*y[] + eps_S[];
        r[]     = phi_pi*pi[] + phi_y*y[] + eps_R[];
        eps_D[] = rho_D*eps_D[-1] + eta_D[];
        eps_S[] = rho_S*eps_S[-1] + eta_S[];
        eps_R[] = rho_R*eps_R[-1] + eta_R[];
    };
    shocks { eta_D[], eta_S[], eta_R[]; };
    calibration {
        beta = ", beta_val, "; sigma = ", sigma_val, ";
        theta = ", theta_val, "; varphi = ", varphi_val, ";
        kappa = ", kappa_val, ";
        phi_pi = 1.5; phi_y = 0.125;
        rho_S = 0.90; rho_D = 0.90; rho_R = 0.40;
    };
};
")

writeLines(model_code_1, "nk_model_standard.gcn")
mod1 <- make_model("nk_model_standard.gcn")
mod1 <- initval_var(mod1, init_var = c(y = 0, pi = 0, r = 0,
                                        eps_D = 0, eps_S = 0, eps_R = 0))
mod1 <- steady_state(mod1)
mod1 <- solve_pert(mod1, loglin = FALSE)
mod1 <- set_shock_distr_par(mod1,
            distr_par = list("sd(eta_D)" = 0.01,
                             "sd(eta_S)" = 0.01,
                             "sd(eta_R)" = 0.01))

cat("\n--- Rozwiazanie perturbacyjne: Standardowa regula Taylora ---\n")
get_pert_solution(mod1)

sim1 <- random_path(mod1, variables = c('y', 'pi', 'r'), sim_length = 100)
plot_simulation(sim1)


# ==============================================================================
# 2. REGULA TAYLORA Z WYGLADZANIEM STOPY PROCENTOWEJ
#    r[] = rho_r * r[-1] + (1 - rho_r) * (phi_pi * pi[] + phi_y * y[]) + eps_R[]
# ==============================================================================

cat("\n\n##############################################\n")
cat("# MODEL 2: Regula Taylora z wygladzaniem    #\n")
cat("##############################################\n\n")

model_code_2 <- paste0("
block EQUILIBRIUM {
    identities {
        y[]     = E[][y[1]] - (1/sigma)*(r[] - E[][pi[1]]) + eps_D[];
        pi[]    = beta*E[][pi[1]] + kappa*y[] + eps_S[];
        r[]     = rho_r*r[-1] + (1 - rho_r)*(phi_pi*pi[] + phi_y*y[]) + eps_R[];
        eps_D[] = rho_D*eps_D[-1] + eta_D[];
        eps_S[] = rho_S*eps_S[-1] + eta_S[];
        eps_R[] = rho_R*eps_R[-1] + eta_R[];
    };
    shocks { eta_D[], eta_S[], eta_R[]; };
    calibration {
        beta = ", beta_val, "; sigma = ", sigma_val, ";
        theta = ", theta_val, "; varphi = ", varphi_val, ";
        kappa = ", kappa_val, ";
        phi_pi = 1.5; phi_y = 0.125;
        rho_r = 0.80;
        rho_S = 0.90; rho_D = 0.90; rho_R = 0.40;
    };
};
")

writeLines(model_code_2, "nk_model_smoothing.gcn")
mod2 <- make_model("nk_model_smoothing.gcn")
mod2 <- initval_var(mod2, init_var = c(y = 0, pi = 0, r = 0,
                                        eps_D = 0, eps_S = 0, eps_R = 0))
mod2 <- steady_state(mod2)
mod2 <- solve_pert(mod2, loglin = FALSE)
mod2 <- set_shock_distr_par(mod2,
            distr_par = list("sd(eta_D)" = 0.01,
                             "sd(eta_S)" = 0.01,
                             "sd(eta_R)" = 0.01))

cat("\n--- Rozwiazanie perturbacyjne: Regula z wygladzaniem (rho_r = 0.80) ---\n")
get_pert_solution(mod2)

sim2 <- random_path(mod2, variables = c('y', 'pi', 'r'), sim_length = 100)
plot_simulation(sim2)


# ==============================================================================
# 3. FORWARD-LOOKING REGULA TAYLORA
#    r[] = phi_pi * E[][pi[1]] + phi_y * E[][y[1]] + eps_R[]
# ==============================================================================

cat("\n\n##############################################\n")
cat("# MODEL 3: Forward-looking regula Taylora   #\n")
cat("##############################################\n\n")

model_code_3 <- paste0("
block EQUILIBRIUM {
    identities {
        y[]     = E[][y[1]] - (1/sigma)*(r[] - E[][pi[1]]) + eps_D[];
        pi[]    = beta*E[][pi[1]] + kappa*y[] + eps_S[];
        r[]     = phi_pi*E[][pi[1]] + phi_y*E[][y[1]] + eps_R[];
        eps_D[] = rho_D*eps_D[-1] + eta_D[];
        eps_S[] = rho_S*eps_S[-1] + eta_S[];
        eps_R[] = rho_R*eps_R[-1] + eta_R[];
    };
    shocks { eta_D[], eta_S[], eta_R[]; };
    calibration {
        beta = ", beta_val, "; sigma = ", sigma_val, ";
        theta = ", theta_val, "; varphi = ", varphi_val, ";
        kappa = ", kappa_val, ";
        phi_pi = 1.5; phi_y = 0.125;
        rho_S = 0.90; rho_D = 0.90; rho_R = 0.40;
    };
};
")

writeLines(model_code_3, "nk_model_forward.gcn")
mod3 <- make_model("nk_model_forward.gcn")
mod3 <- initval_var(mod3, init_var = c(y = 0, pi = 0, r = 0,
                                        eps_D = 0, eps_S = 0, eps_R = 0))
mod3 <- steady_state(mod3)
mod3 <- solve_pert(mod3, loglin = FALSE)
mod3 <- set_shock_distr_par(mod3,
            distr_par = list("sd(eta_D)" = 0.01,
                             "sd(eta_S)" = 0.01,
                             "sd(eta_R)" = 0.01))

cat("\n--- Rozwiazanie perturbacyjne: Forward-looking regula Taylora ---\n")
get_pert_solution(mod3)

sim3 <- random_path(mod3, variables = c('y', 'pi', 'r'), sim_length = 100)
plot_simulation(sim3)


# ==============================================================================
# PODSUMOWANIE
# ==============================================================================

cat("\n\n================================================================\n")
cat("  Rozwiazano 3 warianty modelu NK DSGE:\n")
cat("  1. Standardowa regula Taylora:  r = 1.5*pi + 0.125*y\n")
cat("  2. Z wygladzaniem (rho_r=0.8): r = 0.8*r(-1) + 0.2*(1.5*pi + 0.125*y)\n")
cat("  3. Forward-looking:            r = 1.5*E(pi') + 0.125*E(y')\n")
cat("  Macierze BK wydrukowane powyzej przez get_pert_solution().\n")
cat("================================================================\n")
