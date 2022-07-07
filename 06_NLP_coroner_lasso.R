#Load libraries
library(pacman)
p_load(tidyverse, tidylog, purrr, forcats, visdat, flextable, colorspace, gtsummary, tidymodels, tictoc) #Basic
p_load(textrecipes, tidytext) #Text processing helpers
p_load(parsnip, discrim, naivebayes, ranger, xgboost, kknn, keras, workflowsets, themis, stacks) #Models
doMC::registerDoMC(cores = parallel::detectCores(logical = FALSE)) #Parallel processing
set.seed(100)

#Load data
df <- readRDS("Data/2020 Mortality Data_clean.rds")
df <- df %>% mutate(across(c(13:34), ~as.factor(.))) #Make factors
df <- df %>% mutate(across(c(13:34), ~fct_rev(.))) #Reverse level for proper testing results. 

#Set outcomes list
outcomes.list <- c("Any Opioids", "Heroin", "Fentanyl", "Prescription.opioids", "Methamphetamine", "Cocaine", 
                   "Benzodiazepines", "Alcohol", "Others")
outcomes <- outcomes.list

#Create training/testing dataset
data_split <- map(outcomes.list, ~ 
  initial_split(df, 
                prop = 0.8,
                strata = .x)
)

# Create dataframes for the two sets:
training <- map(data_split, ~ 
                  training(.x)
)
testing <- map(data_split, ~
                 testing(.x)
)

#Load training/testing
training <- read_rds("Data/training.rds")
testing <- read_rds("Data/testing.rds")

#Cross validation
cv_folds <- map2(training, outcomes.list, ~ vfold_cv(.x, v = 10, strata = .y)
)

# Set metrics for classification problem
multi_met <- metric_set(f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)

#Set lists for models
models.list <- c("Lasso") 
recipe.model.list <- c("recipe_Lasso")

#Set-up specs for classifiers
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

#Model over lists
rec.list <- pmap(list(outcomes, outcomes.list, training), ~ 
                  recipe(. ~ text, data = ..3 %>% select(.data[[..1]], text)) %>%
                  #update_role(one_of(.x), new_role = "outcome") %>%
                  update_role(text, new_role = "predictor") %>%
                  step_tokenize(text,
                                options = list(lowercase = TRUE,
                                               strip_punct = TRUE)) %>%
                  step_tokenfilter(text, min_times = 5, max_tokens = 100) %>%
                  step_stopwords(text) %>%
                  step_tfidf(text)
)

#Set-up workflow
wflw_set.list <- map(rec.list, ~ workflow_set(
      preproc = list(.x), 
      models = list(Lasso = lasso_spec)
)
)

#Set up parameter tunning
lambda_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)
lambda_grid

grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = FALSE
   )


#Run workflow - very slow
tic()
grid_results.list <- map2(wflw_set.list, cv_folds, ~
   .x %>%
   workflow_map(
      seed = 100,
      resamples = .y,
      grid = lambda_grid,
      control = grid_ctrl,
      metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv),
      verbose = T
      )
)
time.training <- toc()

#Save/Load to local harddrive
#saveRDS(grid_results.list, paste0("Output/grid_results", as.character(Sys.Date()),"tfidf.rds"))

###Collect metrics
#Create table with all results
results.list <- map2(cross2(1:length(outcomes), recipe.model.list), 
             cross2(outcomes, models.list), 
             ~ {
               extract_workflow_set_result(grid_results.list[[.x[[1]]]], id = .x[[2]]) %>% collect_metrics() %>% mutate(outcome = .y[[1]], model = .y[[2]])
               }
             )
results.tbl.w <- bind_rows(results.list) %>% pivot_wider(
  id_cols = c(.config, outcome, model),
  names_from = .metric,
  values_from = mean
)

#Summary statistics
statistic.plot.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% 
  autoplot() + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Metrics",
       subtitle = paste(.y)) +
    theme_minimal()
})

statistic.best.plot.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% 
  autoplot(select_best = TRUE) + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Metrics",
       subtitle = paste(.y)) +
    theme_minimal()
})

#Rank models
rank_res <- map2(1:length(outcomes), outcomes, ~ { 
  rank_results(grid_results.list[[.x]], 
               rank_metric = "f_meas", 
               select_best = TRUE) %>% mutate(outcome = .y)
})

rank_res.table <- map(1:length(outcomes), ~ {
  rank_res[[.x]] %>% filter(.metric == "f_meas")
})
rank_res.table <- bind_rows(rank_res.table)

#Create table for word
table.train.all.ranked <- flextable::flextable(rank_res.table %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                       select(Substance = outcome, `Model` = model, 
                              `Mean\nF-score` = mean, SE = std_err)) %>%
  set_caption(caption =  "Table. F-score in 10-fold cross validation of training dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.train.all.ranked

#Create ROC curves by model/outcome
roc.list <- map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% collect_predictions() %>%
  group_by(model) %>%
  roc_curve(truth = .data[[.y]], estimate = .pred_1) %>%
  autoplot() +
  labs(
    title = "Receiver operator curve - Training",
    subtitle = paste(.y))
}
)

#Create PR curve by model/outcome
pr.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% collect_predictions() %>%
  group_by(model) %>%
  pr_curve(truth = .data[[y]], .pred_1) %>%
  autoplot() +
  labs(
    title = "Precision-recall curve - Training",
    subtitle = paste(.y))
}
)

#Table of best models
best_results.table <- map(1:length(outcomes), ~ {
  rank_res[[.x]] %>% filter(.metric == "f_meas" & rank < 4)
}
)
best_results.table <- bind_rows(best_results.table)

#Create table for word
table.train.best <- flextable::flextable(best_results.table %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                       select(Substance = outcome, `Model` = model, 
                              `Mean\nF-score` = mean, SE = std_err)) %>%
  set_caption(caption =  "Table. Best performing models in 10-fold cross validation of training dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.train.best

###Set up for testing
#Select best model
best_results.list <- map2(1:length(outcomes), outcomes, ~ { 
   grid_results.list[[.x]] %>% 
   extract_workflow_set_result(rank_res[[.x]]$wflow_id[1]) %>% 
   select_best(metric = "f_meas") %>% mutate(outcome = .y)
}
)

#Fit on test data
tic()
test_results <- map2(1:length(outcomes), data_split, ~ {  
   grid_results.list[[.x]] %>% 
   extract_workflow(rank_res[[.x]]$wflow_id[1]) %>% 
   finalize_workflow(best_results.list[[.x]]) %>% 
   last_fit(split = .y, 
            metrics = metric_set(f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)) 
}
)
time.testing <- toc()

#Save/Load to local harddrive
#saveRDS(test_results, paste0("Output/test_results", as.character(Sys.Date()),"_tfidf.rds"))
#test_results <- readRDS("Output/test_results2021-08-22tfidf.rds")

#Table for test results
test_results.tbl <- map2(1:length(outcomes), outcomes, ~ {  
  test_results[[.x]] %>% collect_metrics() %>% mutate(outcome = .y)
}
)

 test_results.tbl.w <- bind_rows(test_results.tbl) %>% pivot_wider(
  id_cols = c(outcome),
  names_from = .metric,
  values_from = .estimate
)

#Create table for word
table.test <- flextable::flextable(test_results.tbl.w %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                       select(Substance = outcome, `F-score` = `f_meas`, Accuracy = `accuracy`, Kappa = `kap`,
                              `Sensitivity\n(Recall)` = `sens`, Specificity = `spec`,
                              `Positive predictive value\n(Precision)` = `ppv`,
                              `Negative predictive value` = `npv`,
                              AUROC = `roc_auc`)) %>%
  set_caption(caption =  "Table. Diagnostic metrics of best performing models in test dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.test

###Error analysis
#Confusion matrix by model/outcome
test_CM_table <- map2(1:length(outcomes), outcomes, ~ {  
test_results[[.x]] %>% collect_predictions() %>% 
 conf_mat(.y, .pred_class)  %>%
  autoplot(type = "heatmap") +
  labs(
    title = "Confusion matrix",
    subtitle = paste(.y)
  )
})

#Error analysis from test set
error_analysis <- map2(1:length(outcomes), outcomes, ~ {
test_results[[.x]]$.predictions[[1]] %>% #filter(.data[[.y]] != .pred_class) %>% 
    mutate(Error = ifelse(.data[[.y]] != .pred_class,1,0))
})

error_analysis_text <- map(1:length(outcomes), ~ {
  error_analysis[[.x]] %>% left_join(df %>% mutate(.row = row_number()) %>% select(.row, text), by = ".row")
})

#Save excel file with one df per sheet
p_load(openxlsx)
#openxlsx::write.xlsx(error_analysis_text, paste0("Error/error_analysis_df_", Sys.Date(), ".xlsx"))

#Variable importance
p_load(vip)
imp <- map(1:length(outcomes), ~ {
  test_results[[.x]]$.workflow[[1]] %>%
    extract_fit_parsnip() %>%
    vi()
})

test_CM_table <- map2(1:length(outcomes), outcomes, ~ {  
  test_results[[.x]] %>% collect_predictions() %>% 
    conf_mat(.y, .pred_class)  %>%
    autoplot(type = "heatmap") +
    labs(
      title = "Confusion matrix",
      subtitle = paste(.y)
    )
})

p.feature.importance <- map2(1:length(outcomes), outcomes, ~ {
  imp[[.x]] %>%
    mutate(
      Sign = case_when(Sign == "POS" ~ "Negative",
                     Sign == "NEG" ~ "Positive"),
      Importance = abs(Importance),
      Variable = str_remove_all(Variable, "tfidf_text_")
    #Variable = str_remove_all(Variable, "textfeature_narrative_copy_")
  ) %>%
    group_by(Sign) %>%
    top_n(10, Importance) %>%
    ungroup %>%
    ggplot(aes(x = Importance,
             y = fct_reorder(Variable, Importance),
             fill = Sign)) +
    geom_col(show.legend = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    facet_wrap(~Sign, scales = "free") +
    labs(
      y = "Token",
      title = paste("Variable importance for predicting category:", .y)) +
    theme_minimal()
})

###Bootstrapping
#Confidence intervals
boot_test <- map(testing, ~ bootstraps(.x, times = 1000, apparent = TRUE)
)

tic()
bootstrap_results <- map2(1:length(outcomes), boot_test, ~ {  
   grid_results.list[[.x]] %>% 
   extract_workflow(rank_res[[.x]]$wflow_id[1]) %>% 
   finalize_workflow(best_results.list[[.x]]) %>% 
   fit_resamples(resamples = .y,
            metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)) 
}
)
time.bootstrap <- toc()

#Save/Load to local harddrive
#saveRDS(bootstrap_results, paste0("Output/bootstrap_results", as.character(Sys.Date()),"_tfidf.rds"))

bootstrap_CI_results <- map2(1:length(outcomes), outcomes, ~ {
  bootstrap_results[[.x]] %>% collect_metrics(summarize = FALSE) %>%
    group_by(.metric) %>% 
    summarize(mean = mean(.estimate, na.rm = T),
                                  CI.lower = quantile(.estimate, probs = c(0.025), na.rm = T),
                                  CI.upper = quantile(.estimate, probs = c(0.975), na.rm = T)
                                  ) %>%
    mutate(outcome = .y)
})

CI_results.tbl.w <- bind_rows(bootstrap_CI_results) %>% pivot_wider(
  id_cols = c(outcome),
  names_from = .metric,
  values_from = c(mean, CI.lower, CI.upper)
)

table.95CI <- flextable::flextable(CI_results.tbl.w %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
  select(Substance = outcome, 
         `F-score` = mean_f_meas, Lower = CI.lower_f_meas, Upper = CI.upper_f_meas,
         `Accuracy` = mean_accuracy, aLower = CI.lower_accuracy, aUpper = CI.upper_accuracy,
         `Kappa` = mean_kap, kLower = CI.lower_kap, kUpper = CI.upper_kap, 
         `Sensitivity\n(Recall)` = `mean_sens`, sLower = CI.lower_sens, sUpper = CI.upper_sens,
         Specificity = `mean_spec`, spLower = CI.lower_spec, spUpper = CI.upper_spec,
         `Positive predictive value\n(Precision)` = `mean_ppv`, ppLower = CI.lower_ppv, ppUpper = CI.upper_ppv,
         `Negative predictive value` = `mean_npv`, npLower = CI.lower_npv, npUpper = CI.upper_npv,
         AUROC = `mean_roc_auc`, auLower = CI.lower_roc_auc, auUpper = CI.upper_roc_auc)) %>%
  set_caption(caption = "Table. Bootstrapped diagnostic metrics and 95% confidence intervals of best performing models in test dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.95CI

###Export to word
#Final tables/plots
tables.list <- list(table.train.all.ranked, table.train.best, table.test, table.95CI)

# write function
p_load(officer, flextable)
write_word_table <- function(var, doc){
  doc %>%
    body_add_flextable(var) %>% 
    body_add_break() }

write_word_plot <- function(var, doc){
  doc %>%
    body_add_gg(var, style = "centered", height = 6, width = 8) %>% 
    body_end_section_landscape()
}

# list of tables and the doc
my_doc1 <- officer::read_docx()

# use walk (the invisible function of map) to include all tables in one doc
walk(tables.list, write_word_table, my_doc1) 

# use walk to include plots
walk(statistic.plot.list, write_word_plot, my_doc1) 
walk(statistic.best.plot.list, write_word_plot, my_doc1) 
walk(test_CM_table, write_word_plot, my_doc1) 
walk(p.feature.importance, write_word_plot, my_doc1) 

#Create word doc
print(my_doc1, target = paste0("Tables/NLP_tables_", Sys.Date(), "_tf_idf_lasso.docx")) %>% invisible()

#################### Glove analysis

#Set outcomes list
outcomes.list <- c("Any Opioids", "Heroin", "Fentanyl", "Prescription.opioids", "Methamphetamine", "Cocaine", 
                   "Benzodiazepines", "Alcohol", "Others")
outcomes <- outcomes.list

#Download GloVe embeddings
glove6b <- textdata::embedding_glove6b(dimensions = 100)

#Set lists for models
models.list <- c("Lasso") 
recipe.model.list <- c("recipe_Lasso")

#Set-up specs for classifiers
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")


#Model over lists
rec.list <- pmap(list(outcomes, outcomes.list, training), ~ 
                   recipe(. ~ text, data = ..3 %>% select(.data[[..1]], text)) %>%
                   #update_role(one_of(.x), new_role = "outcome") %>%
                   update_role(text, new_role = "predictor") %>%
                   step_tokenize(text,
                                 options = list(lowercase = F,
                                                strip_punct = TRUE)) %>%
                   step_tokenfilter(text, min_times = 5) %>%       
                   step_stopwords(text) %>%
                   step_word_embeddings(text, embeddings = glove6b)
                   
)

#Set-up workflow
wflw_set.list <- map(rec.list, ~ workflow_set(
  preproc = list(.x), 
  models = list(Lasso = lasso_spec)
)
)

#Set up parameter tunning
lambda_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)
lambda_grid

grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = FALSE
  )


#Run workflow - very slow
tic()
grid_results.list <- map2(wflw_set.list, cv_folds, ~
                            .x %>%
                            workflow_map(
                              seed = 100,
                              resamples = .y,
                              grid = lambda_grid,
                              control = grid_ctrl,
                              metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv),
                              verbose = T
                            )
)
time.training <- toc()

#Save/Load to local harddrive
#saveRDS(grid_results.list, paste0("Output/grid_results", as.character(Sys.Date()),"glove.rds"))

###Collect metrics
#Create table with all results
results.list <- map2(cross2(1:length(outcomes), recipe.model.list), 
                     cross2(outcomes, models.list), 
                     ~ {
                       extract_workflow_set_result(grid_results.list[[.x[[1]]]], id = .x[[2]]) %>% collect_metrics() %>% mutate(outcome = .y[[1]], model = .y[[2]])
                     }
)
results.tbl.w <- bind_rows(results.list) %>% pivot_wider(
  id_cols = c(.config, outcome, model),
  names_from = .metric,
  values_from = mean
)

#Summary statistics
statistic.plot.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% 
    autoplot() + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Metrics",
         subtitle = paste(.y)) +
    theme_minimal()
})

statistic.best.plot.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% 
    autoplot(select_best = TRUE) + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Metrics",
         subtitle = paste(.y)) +
    theme_minimal()
})

#Rank models
rank_res <- map2(1:length(outcomes), outcomes, ~ { 
  rank_results(grid_results.list[[.x]], 
               rank_metric = "f_meas", 
               select_best = TRUE) %>% mutate(outcome = .y)
})

rank_res.table <- map(1:length(outcomes), ~ {
  rank_res[[.x]] %>% filter(.metric == "f_meas")
})
rank_res.table <- bind_rows(rank_res.table)

#Create table for word
table.train.all.ranked <- flextable::flextable(rank_res.table %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                                 select(Substance = outcome, `Model` = model, 
                                                        `Mean\nF-score` = mean, SE = std_err)) %>%
  set_caption(caption =  "Table. F-score in 10-fold cross validation of training dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.train.all.ranked

#Create ROC curves by model/outcome
roc.list <- map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% collect_predictions() %>%
    group_by(model) %>%
    roc_curve(truth = .data[[.y]], estimate = .pred_1) %>%
    autoplot() +
    labs(
      title = "Receiver operator curve - Training",
      subtitle = paste(.y))
}
)

#Create PR curve by model/outcome
pr.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% collect_predictions() %>%
    group_by(model) %>%
    pr_curve(truth = .data[[y]], .pred_1) %>%
    autoplot() +
    labs(
      title = "Precision-recall curve - Training",
      subtitle = paste(.y))
}
)

#Table of best models
best_results.table <- map(1:length(outcomes), ~ {
  rank_res[[.x]] %>% filter(.metric == "f_meas" & rank < 4)
}
)
best_results.table <- bind_rows(best_results.table)

#Create table for word
table.train.best <- flextable::flextable(best_results.table %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                           select(Substance = outcome, `Model` = model, 
                                                  `Mean\nF-score` = mean, SE = std_err)) %>%
  set_caption(caption =  "Table. Best performing models in 10-fold cross validation of training dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.train.best

###Set up for testing
#Select best model
best_results.list <- map2(1:length(outcomes), outcomes, ~ { 
  grid_results.list[[.x]] %>% 
    extract_workflow_set_result(rank_res[[.x]]$wflow_id[1]) %>% 
    select_best(metric = "f_meas") %>% mutate(outcome = .y)
}
)

#Fit on test data
tic()
test_results <- map2(1:length(outcomes), data_split, ~ {  
  grid_results.list[[.x]] %>% 
    extract_workflow(rank_res[[.x]]$wflow_id[1]) %>% 
    finalize_workflow(best_results.list[[.x]]) %>% 
    last_fit(split = .y, 
             metrics = metric_set(f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)) 
}
)
time.testing <- toc()

#Save/Load to local harddrive
#saveRDS(test_results, paste0("Output/test_results", as.character(Sys.Date()),"_glove.rds"))
#test_results <- readRDS("Output/test_results2021-08-22glove.rds")

#Table for test results
test_results.tbl <- map2(1:length(outcomes), outcomes, ~ {  
  test_results[[.x]] %>% collect_metrics() %>% mutate(outcome = .y)
}
)

test_results.tbl.w <- bind_rows(test_results.tbl) %>% pivot_wider(
  id_cols = c(outcome),
  names_from = .metric,
  values_from = .estimate
)

#Create table for word
table.test <- flextable::flextable(test_results.tbl.w %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                     select(Substance = outcome, `F-score` = `f_meas`, Accuracy = `accuracy`, Kappa = `kap`,
                                            `Sensitivity\n(Recall)` = `sens`, Specificity = `spec`,
                                            `Positive predictive value\n(Precision)` = `ppv`,
                                            `Negative predictive value` = `npv`,
                                            AUROC = `roc_auc`)) %>%
  set_caption(caption =  "Table. Diagnostic metrics of best performing models in test dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.test

###Error analysis
#Confusion matrix by model/outcome
test_CM_table <- map2(1:length(outcomes), outcomes, ~ {  
  test_results[[.x]] %>% collect_predictions() %>% 
    conf_mat(.y, .pred_class)  %>%
    autoplot(type = "heatmap") +
    labs(
      title = "Confusion matrix",
      subtitle = paste(.y)
    )
})

#Error analysis from test set
error_analysis <- map2(1:length(outcomes), outcomes, ~ {
  test_results[[.x]]$.predictions[[1]] %>% #filter(.data[[.y]] != .pred_class) %>% 
    mutate(Error = ifelse(.data[[.y]] != .pred_class,1,0))
})

error_analysis_text <- map(1:length(outcomes), ~ {
  error_analysis[[.x]] %>% left_join(df %>% mutate(.row = row_number()) %>% select(.row, text), by = ".row")
})

#Save excel file with one df per sheet
p_load(openxlsx)
#openxlsx::write.xlsx(error_analysis_text, paste0("Error/error_analysis_df_glove", Sys.Date(), ".xlsx"))

#Feature importance
imp <- map(1:length(outcomes), ~ {
  test_results[[.x]]$.workflow[[1]] %>%
    extract_fit_parsnip() %>%
    vi()
})

p.feature.importance <- map2(1:length(outcomes), outcomes, ~ {
  imp[[.x]] %>%
    mutate(
      Sign = case_when(Sign == "POS" ~ "Negative",
                       Sign == "NEG" ~ "Positive"),
      Importance = abs(Importance),
      Variable = str_remove_all(Variable, "tfidf_text_")
      #Variable = str_remove_all(Variable, "textfeature_narrative_copy_")
    ) %>%
    group_by(Sign) %>%
    top_n(10, Importance) %>%
    ungroup %>%
    ggplot(aes(x = Importance,
               y = fct_reorder(Variable, Importance),
               fill = Sign)) +
    geom_col(show.legend = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    facet_wrap(~Sign, scales = "free") +
    labs(
      y = "Token",
      title = paste("Variable importance for predicting category:", .y)) +
    theme_minimal()
})

###Bootstrapping
#Confidence intervals
boot_test <- map(testing, ~ bootstraps(.x, times = 1000, apparent = TRUE)
)

tic()
bootstrap_results <- map2(1:length(outcomes), boot_test, ~ {  
  grid_results.list[[.x]] %>% 
    extract_workflow(rank_res[[.x]]$wflow_id[1]) %>% 
    finalize_workflow(best_results.list[[.x]]) %>% 
    fit_resamples(resamples = .y,
                  metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)) 
}
)
time.bootstrap <- toc()

#Save/Load to local harddrive
#saveRDS(bootstrap_results, paste0("Output/bootstrap_results", as.character(Sys.Date()),"_Glove.rds"))

bootstrap_CI_results <- map2(1:length(outcomes), outcomes, ~ {
  bootstrap_results[[.x]] %>% collect_metrics(summarize = FALSE) %>%
    group_by(.metric) %>% 
    summarize(mean = mean(.estimate, na.rm = T),
              CI.lower = quantile(.estimate, probs = c(0.025), na.rm = T),
              CI.upper = quantile(.estimate, probs = c(0.975), na.rm = T)
    ) %>%
    mutate(outcome = .y)
})

CI_results.tbl.w <- bind_rows(bootstrap_CI_results) %>% pivot_wider(
  id_cols = c(outcome),
  names_from = .metric,
  values_from = c(mean, CI.lower, CI.upper)
)

table.95CI <- flextable::flextable(CI_results.tbl.w %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                     select(Substance = outcome, 
                                            `F-score` = mean_f_meas, Lower = CI.lower_f_meas, Upper = CI.upper_f_meas,
                                            `Accuracy` = mean_accuracy, aLower = CI.lower_accuracy, aUpper = CI.upper_accuracy,
                                            `Kappa` = mean_kap, kLower = CI.lower_kap, kUpper = CI.upper_kap, 
                                            `Sensitivity\n(Recall)` = `mean_sens`, sLower = CI.lower_sens, sUpper = CI.upper_sens,
                                            Specificity = `mean_spec`, spLower = CI.lower_spec, spUpper = CI.upper_spec,
                                            `Positive predictive value\n(Precision)` = `mean_ppv`, ppLower = CI.lower_ppv, ppUpper = CI.upper_ppv,
                                            `Negative predictive value` = `mean_npv`, npLower = CI.lower_npv, npUpper = CI.upper_npv,
                                            AUROC = `mean_roc_auc`, auLower = CI.lower_roc_auc, auUpper = CI.upper_roc_auc)) %>%
  set_caption(caption = "Table. Bootstrapped diagnostic metrics and 95% confidence intervals of best performing models in test dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.95CI

###Export to word
#Final tables/plots
tables.list <- list(table.train.all.ranked, table.train.best, table.test, table.95CI)



# write function
p_load(officer, flextable)
write_word_table <- function(var, doc){
  doc %>%
    body_add_flextable(var) %>% 
    body_add_break() }

write_word_plot <- function(var, doc){
  doc %>%
    body_add_gg(var, style = "centered", height = 6, width = 8) %>% 
    body_end_section_landscape()
}
# list of tables and the doc
my_doc1 <- officer::read_docx()

# use walk (the invisible function of map) to include all tables in one doc
walk(tables.list, write_word_table, my_doc1) 

# use walk to include plots
walk(statistic.plot.list, write_word_plot, my_doc1) 
walk(statistic.best.plot.list, write_word_plot, my_doc1) 
walk(test_CM_table, write_word_plot, my_doc1) 
walk(p.feature.importance, write_word_plot, my_doc1) 

#Create word doc
print(my_doc1, target = paste0("Tables/NLP_tables_", Sys.Date(), "_Glove_lasso.docx")) %>% invisible()

########################Cui2vec
### Load Clinspacy
# https://github.com/ML4LHS/clinspacy
# reticulate::conda_remove('clinspacy')
# remotes::install_github('ML4LHS/clinspacy', force = T)
pacman::p_load(clinspacy)

# Initiate clinspacy / use UMLS linker (this may take a very long time)
#clinspacy_init(use_linker = TRUE)

# Load cui2vec
#cui2vec_definitions <- dataset_cui2vec_definitions()

#cui2vec_embed <- dataset_cui2vec_embeddings()
#cui2vec_embed <- cui2vec_embed %>% rename(token = cui)
#cui2vec_embed <- as_tibble(cui2vec_embed)
#saveRDS(cui2vec_embed, "Data/cui2vec_embed.RDS")
cui2vec_embed <- readRDS("Data/cui2vec_embed.RDS")
cui2vec_embed <- as_tibble(cui2vec_embed)

#Load already transformed main df
df.cui2 <- read_rds("Data/df_cui2.RDS")

#Set outcomes list
outcomes.list <- c("Any Opioids", "Heroin", "Fentanyl", "Prescription.opioids", "Methamphetamine", "Cocaine", 
                   "Benzodiazepines", "Alcohol", "Others")
outcomes <- outcomes.list

#Create training/testing dataset
data_split <- map(outcomes.list, ~ 
                    initial_split(df.cui2, 
                                  prop = 0.8,
                                  strata = .x)
)

# Create dataframes for the two sets:
training <- map(data_split, ~ 
                  training(.x)
)
testing <- map(data_split, ~
                 testing(.x)
)

#Cross validation
cv_folds <- map2(training, outcomes.list, ~ vfold_cv(.x, v = 10, strata = .y)
)

#Set lists for models
models.list <- c("Lasso") 
recipe.model.list <- c("recipe_Lasso")

#Set-up specs for classifiers
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

#Model over lists
rec.list <- pmap(list(outcomes, outcomes.list, training), ~ 
                   recipe(. ~ cui, data = ..3 %>% select(.data[[..1]], cui)) %>%
                   #update_role(one_of(.x), new_role = "outcome") %>%
                   update_role(cui, new_role = "predictor") %>%
                   step_tokenize(cui,
                                 options = list(lowercase = F,
                                                strip_punct = TRUE)) %>%
                   step_tokenfilter(cui, min_times = 5) %>%
                   step_word_embeddings(cui, embeddings = cui2vec_embed, prefix = "emb_")
)


#Set-up workflow
wflw_set.list <- map(rec.list, ~ workflow_set(
  preproc = list(.x), 
  models = list(Lasso = lasso_spec)
)
)

#Set up grid control
grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = FALSE
  )

#Set up parameter tunning
lambda_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)
lambda_grid


#Run workflow - very slow
tic()
grid_results.list <- map2(wflw_set.list, cv_folds, ~
                            .x %>%
                            workflow_map(
                              seed = 100,
                              resamples = .y,
                              grid = lambda_grid,
                              control = grid_ctrl,
                              metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv),
                              verbose = T
                            )
)
time.training <- toc()

#Save/Load to local harddrive
#saveRDS(grid_results.list, paste0("Output/grid_results", as.character(Sys.Date()),"cui2vec.rds"))

###Collect metrics
#Create table with all results
results.list <- map2(cross2(1:length(outcomes), recipe.model.list), 
                     cross2(outcomes, models.list), 
                     ~ {
                       extract_workflow_set_result(grid_results.list[[.x[[1]]]], id = .x[[2]]) %>% collect_metrics() %>% mutate(outcome = .y[[1]], model = .y[[2]])
                     }
)
results.tbl.w <- bind_rows(results.list) %>% pivot_wider(
  id_cols = c(.config, outcome, model),
  names_from = .metric,
  values_from = mean
)

#Summary statistics
statistic.plot.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% 
    autoplot() + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Metrics",
         subtitle = paste(.y)) +
    theme_minimal()
})

statistic.best.plot.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% 
    autoplot(select_best = TRUE) + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "Metrics",
         subtitle = paste(.y)) +
    theme_minimal()
})

#Rank models
rank_res <- map2(1:length(outcomes), outcomes, ~ { 
  rank_results(grid_results.list[[.x]], 
               rank_metric = "f_meas", 
               select_best = TRUE) %>% mutate(outcome = .y)
})

rank_res.table <- map(1:length(outcomes), ~ {
  rank_res[[.x]] %>% filter(.metric == "f_meas")
})
rank_res.table <- bind_rows(rank_res.table)

#Create table for word
table.train.all.ranked <- flextable::flextable(rank_res.table %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                                 select(Substance = outcome, `Model` = model, 
                                                        `Mean\nF-score` = mean, SE = std_err)) %>%
  set_caption(caption =  "Table. F-score in 10-fold cross validation of training dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.train.all.ranked

#Create ROC curves by model/outcome
roc.list <- map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% collect_predictions() %>%
    group_by(model) %>%
    roc_curve(truth = .data[[.y]], estimate = .pred_1) %>%
    autoplot() +
    labs(
      title = "Receiver operator curve - Training",
      subtitle = paste(.y))
}
)

#Create PR curve by model/outcome
pr.list <-  map2(1:length(outcomes), outcomes, ~ {
  grid_results.list[[.x]] %>% collect_predictions() %>%
    group_by(model) %>%
    pr_curve(truth = .data[[y]], .pred_1) %>%
    autoplot() +
    labs(
      title = "Precision-recall curve - Training",
      subtitle = paste(.y))
}
)

#Table of best models
best_results.table <- map(1:length(outcomes), ~ {
  rank_res[[.x]] %>% filter(.metric == "f_meas" & rank < 4)
}
)
best_results.table <- bind_rows(best_results.table)

#Create table for word
table.train.best <- flextable::flextable(best_results.table %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                           select(Substance = outcome, `Model` = model, 
                                                  `Mean\nF-score` = mean, SE = std_err)) %>%
  set_caption(caption =  "Table. Best performing models in 10-fold cross validation of training dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.train.best

###Set up for testing
#Select best model
best_results.list <- map2(1:length(outcomes), outcomes, ~ { 
  grid_results.list[[.x]] %>% 
    extract_workflow_set_result(rank_res[[.x]]$wflow_id[1]) %>% 
    select_best(metric = "f_meas") %>% mutate(outcome = .y)
}
)

#Fit on test data
tic()
test_results <- map2(1:length(outcomes), data_split, ~ {  
  grid_results.list[[.x]] %>% 
    extract_workflow(rank_res[[.x]]$wflow_id[1]) %>% 
    finalize_workflow(best_results.list[[.x]]) %>% 
    last_fit(split = .y, 
             metrics = metric_set(f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)) 
}
)
time.testing <- toc()

#Save/Load to local harddrive
#saveRDS(test_results, paste0("Output/test_results", as.character(Sys.Date()),"_cui2vec.rds"))
#test_results <- readRDS("Output/test_results2021-08-22cui2vec.rds")

#Table for test results
test_results.tbl <- map2(1:length(outcomes), outcomes, ~ {  
  test_results[[.x]] %>% collect_metrics() %>% mutate(outcome = .y)
}
)

test_results.tbl.w <- bind_rows(test_results.tbl) %>% pivot_wider(
  id_cols = c(outcome),
  names_from = .metric,
  values_from = .estimate
)

#Create table for word
table.test <- flextable::flextable(test_results.tbl.w %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                     select(Substance = outcome, `F-score` = `f_meas`, Accuracy = `accuracy`, Kappa = `kap`,
                                            `Sensitivity\n(Recall)` = `sens`, Specificity = `spec`,
                                            `Positive predictive value\n(Precision)` = `ppv`,
                                            `Negative predictive value` = `npv`,
                                            AUROC = `roc_auc`)) %>%
  set_caption(caption =  "Table. Diagnostic metrics of best performing models in test dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.test

###Error analysis
#Confusion matrix by model/outcome
test_CM_table <- map2(1:length(outcomes), outcomes, ~ {  
  test_results[[.x]] %>% collect_predictions() %>% 
    conf_mat(.y, .pred_class)  %>%
    autoplot(type = "heatmap") +
    labs(
      title = "Confusion matrix",
      subtitle = paste(.y)
    )
})

#Error analysis from test set
error_analysis <- map2(1:length(outcomes), outcomes, ~ {
  test_results[[.x]]$.predictions[[1]] %>% #filter(.data[[.y]] != .pred_class) %>% 
    mutate(Error = ifelse(.data[[.y]] != .pred_class,1,0))
})

error_analysis_text <- map(1:length(outcomes), ~ {
  error_analysis[[.x]] %>% left_join(df %>% mutate(.row = row_number()) %>% select(.row, text), by = ".row")
})

#Save excel file with one df per sheet
p_load(openxlsx)
#openxlsx::write.xlsx(error_analysis_text, paste0("Error/error_analysis_df_cui2vec", Sys.Date(), ".xlsx"))

#Variable importance
p_load(vip)
imp <- map(1:length(outcomes), ~ {
  test_results[[.x]]$.workflow[[1]] %>%
    extract_fit_parsnip() %>%
    vi()
})

p.feature.importance <- map2(1:length(outcomes), outcomes, ~ {
  imp[[.x]] %>%
    mutate(
      Sign = case_when(Sign == "POS" ~ "Negative",
                       Sign == "NEG" ~ "Positive"),
      Importance = abs(Importance),
      Variable = str_remove_all(Variable, "tfidf_text_")
      #Variable = str_remove_all(Variable, "textfeature_narrative_copy_")
    ) %>%
    group_by(Sign) %>%
    top_n(10, Importance) %>%
    ungroup %>%
    ggplot(aes(x = Importance,
               y = fct_reorder(Variable, Importance),
               fill = Sign)) +
    geom_col(show.legend = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    facet_wrap(~Sign, scales = "free") +
    labs(
      y = "Token",
      title = paste("Variable importance for predicting category:", .y)) +
    theme_minimal()
})

###Bootstrapping
#Confidence intervals
boot_test <- map(testing, ~ bootstraps(.x, times = 1000, apparent = TRUE)
)

tic()
bootstrap_results <- map2(1:length(outcomes), boot_test, ~ {  
  grid_results.list[[.x]] %>% 
    extract_workflow(rank_res[[.x]]$wflow_id[1]) %>% 
    finalize_workflow(best_results.list[[.x]]) %>% 
    fit_resamples(resamples = .y,
                  metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)) 
}
)
time.bootstrap <- toc()

#Save/Load to local harddrive
#saveRDS(bootstrap_results, paste0("Output/bootstrap_results", as.character(Sys.Date()),"_cui2vec.rds"))

bootstrap_CI_results <- map2(1:length(outcomes), outcomes, ~ {
  bootstrap_results[[.x]] %>% collect_metrics(summarize = FALSE) %>%
    group_by(.metric) %>% 
    summarize(mean = mean(.estimate, na.rm = T),
              CI.lower = quantile(.estimate, probs = c(0.025), na.rm = T),
              CI.upper = quantile(.estimate, probs = c(0.975), na.rm = T)
    ) %>%
    mutate(outcome = .y)
})

CI_results.tbl.w <- bind_rows(bootstrap_CI_results) %>% pivot_wider(
  id_cols = c(outcome),
  names_from = .metric,
  values_from = c(mean, CI.lower, CI.upper)
)

table.95CI <- flextable::flextable(CI_results.tbl.w %>% mutate(across(where(is.numeric), ~round(.x, 3))) %>%
                                     select(Substance = outcome, 
                                            `F-score` = mean_f_meas, Lower = CI.lower_f_meas, Upper = CI.upper_f_meas,
                                            `Accuracy` = mean_accuracy, aLower = CI.lower_accuracy, aUpper = CI.upper_accuracy,
                                            `Kappa` = mean_kap, kLower = CI.lower_kap, kUpper = CI.upper_kap, 
                                            `Sensitivity\n(Recall)` = `mean_sens`, sLower = CI.lower_sens, sUpper = CI.upper_sens,
                                            Specificity = `mean_spec`, spLower = CI.lower_spec, spUpper = CI.upper_spec,
                                            `Positive predictive value\n(Precision)` = `mean_ppv`, ppLower = CI.lower_ppv, ppUpper = CI.upper_ppv,
                                            `Negative predictive value` = `mean_npv`, npLower = CI.lower_npv, npUpper = CI.upper_npv,
                                            AUROC = `mean_roc_auc`, auLower = CI.lower_roc_auc, auUpper = CI.upper_roc_auc)) %>%
  set_caption(caption = "Table. Bootstrapped diagnostic metrics and 95% confidence intervals of best performing models in test dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
table.95CI

###Export to word
#Final tables/plots
tables.list <- list(table.train.all.ranked, table.train.best, table.test, table.95CI)



# write function
p_load(officer, flextable)
write_word_table <- function(var, doc){
  doc %>%
    body_add_flextable(var) %>% 
    body_add_break() }

write_word_plot <- function(var, doc){
  doc %>%
    body_add_gg(var, style = "centered", height = 6, width = 8) %>% 
    body_end_section_landscape()
}
# list of tables and the doc
my_doc1 <- officer::read_docx()

# use walk (the invisible function of map) to include all tables in one doc
walk(tables.list, write_word_table, my_doc1) 

# use walk to include plots
walk(statistic.plot.list, write_word_plot, my_doc1) 
walk(statistic.best.plot.list, write_word_plot, my_doc1) 
walk(test_CM_table, write_word_plot, my_doc1) 
walk(p.feature.importance, write_word_plot, my_doc1) 

#Create word doc
print(my_doc1, target = paste0("Tables/NLP_tables_", Sys.Date(), "_cui2vec_lasso.docx")) %>% invisible()

