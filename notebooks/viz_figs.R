library(tidyverse)
library(ggpattern)
library(stargazer)
extrafont::loadfonts(quiet = T)

global_theme <- function(){
  
  theme_minimal() %+replace%
    theme(
      text=element_text(family='Montserrat', size=14),
      axis.text = element_text(size=14), 
      plot.title = element_text(family='Montserrat SemiBold', size=18, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
}

setwd('~/Documents/Uni/PhD/UQAM/courses/RL/mcgill_precup/project/')

#### Viz Replication ####

data_replication <- read_csv('data/results/replication_calvano.csv')

data_replication %>% 
  ungroup() %>% 
  mutate(epsilon_decay = epsilon_decay*1e5) %>% 
  ggplot() + 
  geom_tile(aes(x=epsilon_decay, y=lr, fill=profit_delta)) +   
  scale_fill_gradient2(low='#FCF4EB', mid='#ff9000', high = '#990f50', 
                       midpoint = 0.55,
                       breaks=seq(0.3,0.8,0.1)) + 
  global_theme() + 
  xlab(expression(beta %*% 10^-5)) + 
  ylab(expression(alpha)) + 
  guides(fill=guide_legend(title=expression(paste(Delta, '-Profit')),
                           reverse = T))

#### DQN Comparison ####

read_csv('data/results/calvano_simple_tabular.csv') %>% 
  rowid_to_column('timestep') -> training_q_sims
data_single <- read_csv('data/results/agent_1_dqn.csv')
data_multi <- read_csv('data/results/agent_1_multi_dqn.csv')


colors <- c("Tabular-Q" = "red",
            "DQN" = "blue", 
             "Multi-State-DQN" = "purple")

ggplot() + 
  geom_line(aes(x=timestep,
                y=mean_1,
                color='Tabular-Q'),
            data=training_q_sims, alpha=0.1) +
  geom_line(aes(x=timestep,
                y=value,
                color='DQN'),
            data=data_single,
            alpha=0.5) + 
  geom_line(aes(x=timestep,
                y=value,
                color='Multi-State-DQN'),
            data=data_multi, alpha=0.5) +
  global_theme() + 
  scale_color_manual(values = colors) + 
  guides(color=guide_legend(title="Algorithm")) + 
  xlab('Timestep') + 
  ylab('Mean profit') + 
  xlim(c(0,4e05)) +
  geom_hline(yintercept = 0.2229281, lty='longdash') + 
  geom_hline(yintercept = 0.3374905, lty='longdash')

#### Multi-Agent ####

data_agent1_multi <- read_csv('data/results/agent_1_multi_dqn_results.csv')
data_agent2_multi <- read_csv('data/results/agent_2_multi_dqn_results.csv')
data_agent3_multi <-read_csv('data/results/agent_3_multi_dqn_results.csv')

ggplot() + 
  geom_line(aes(x=timestep,
                y=agent_2_mean),
            color='blue',
            data=data_agent2_multi, alpha=0.8) + 
  geom_line(aes(x=timestep,
                y=agent_3_mean),
            color='lightgreen',
            data=data_agent3_multi, alpha=0.5) + 
  geom_line(aes(x=timestep,
                y=agent_1_mean),
            color='purple',
            data=data_agent1_multi, alpha=0.5) + 
  global_theme() +
  xlab('Timestep') + 
  ylab('Mean profit') + 
  xlim(c(0,4.5e5)) +
  geom_hline(yintercept = 0.0615224459746, lty='longdash')

#### DDGP ####

data_continous <- read_csv('data/results/continous_results.csv')
data_continous %>% 
  rowid_to_column('timestep') -> data_continous

colors <- c("Tabular-Q" = "red",
            "DQN" = "blue", 
            "DDGP" = "purple")

data_continous %>% 
  select(timestep, mean_1, mean_2) %>% 
  gather(agent, value, c(mean_1:mean_2)) %>% 
  mutate(
    Agent = if_else(agent == 'mean_1', 'Agent 1', 'Agent 2')
  ) %>% 
  ggplot() + 
  geom_line(aes(x=timestep,
                y=value, 
                color=Agent)) + 
  global_theme() + 
  scale_color_manual(values=c('red', 'blue')) +
  xlab('Timestep') + 
  ylab('Mean profit') + 
  geom_hline(yintercept = 0.3374905, lty='longdash')

#### Viz Equilibirium ####
data_monopoly <- read_csv('data/results/price_dynamics_monopoly.csv')

data_monopoly %>% 
  gather(agent, profit, c(profit_2:profit_1)) %>% 
  mutate(
    Agent = if_else(agent == 'profit_1', 'Agent 1', 'Agent 2')
  ) %>% 
  ggplot() + 
  geom_line(aes(x=price_prop, y=profit, color=Agent)) + 
  geom_point(aes(x=1.901, y=0.337), size=2) + 
  geom_point(aes(x=1.6666603333333334, y=0.40430998266850926), size=2) + 
  geom_label(aes(x=1.78, y=0.337, label='Monopoly Equilibirium')) + 
  geom_label(aes(x=1.58, y=0.40430998266850926, label='Best Response')) + 
  global_theme() + 
  scale_color_manual(values=c('red', 'blue')) +
  xlab('Price Proposition') + 
  ylab('Profit')

data_duopoly <- read_csv('data/results/price_dynamics_duopoly.csv')

data_duopoly %>% 
  gather(agent, profit, c(profit_2:profit_1)) %>% 
  mutate(
    Agent = if_else(agent == 'profit_1', 'Agent 1', 'Agent 2')
  ) %>% 
  ggplot() + 
  geom_line(aes(x=price_prop, y=profit, color=Agent)) + 
  geom_point(aes(x=1.4878323333333334, y=0.22914694176000214), size=2)  + 
  geom_label(aes(x=1.63, y=0.22914694176000214, label='Duopoly Equilibirium')) + 
  global_theme() + 
  scale_color_manual(values=c('red', 'blue')) +
  xlab('Price Proposition') + 
  ylab('Profit')


# Table 
data_continous %>% tail(1000) %>% 
  pull(mean_1) %>% 
  mean() -> mean_continous

data_single %>% 
  tail(1000) %>% 
  pull(value) %>% 
  mean() -> mean_single_case

data_multi %>% 
  tail(1000) %>% 
  pull(value) %>% 
  mean() -> mean_multi_case

training_q_sims %>% 
  tail(1000) %>% 
  pull(mean_1) %>% 
  mean() -> mean_tabular

# std
data_continous %>% tail(1000) %>%
  pull(std_1) %>% 
  mean() -> std_continous

data_single %>% 
  tail(1000) %>% 
  pull(value) %>% 
  var() %>% sqrt() -> std_single_case

data_multi %>% 
  tail(1000) %>% 
  pull(value) %>% 
  var() %>% sqrt() -> std_multi_case

training_q_sims %>% 
  tail(1000) %>% 
  pull(std_1) %>% 
  mean() -> std_tabular

mean_vals = c(mean_tabular, mean_single_case, mean_multi_case, mean_continous)
algorithm = c('Tabular', 'DQN-Single', 'DQN-Multi', 'DDPG')

tibble(
  Algorithm = algorithm,
  mean_values = mean_vals, 
) %>% 
  stargazer(summary = F)


####

sample_1 <- rgamma(100000, shape = 2, scale = 1)

qgamma(0.95, shape = 2, scale = 1)
quantile(sample_1, 0.95)

w <- 2
sample_2 <- sample_1
sample_2[sample_2 > w] <- w
sample_1[sample_1 > w] %>% mean()

sample_3 <- sample_1
sample_3[sample_3 < w] <- 0

t1 <- sample_2 + mean(sample_1[sample_1 > w])
tr <- sample_3 - mean(sample_1[sample_1 > w])

quantile(t1, 0.95)
quantile(tr, 0.95)


sin(quantile(sample_1, 0.95))
quantile(sin(sample_1), 0.95)
