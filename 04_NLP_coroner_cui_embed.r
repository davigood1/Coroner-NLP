Copyright (c) 2022, David Goodman-Meza
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 

#Load libraries
library(pacman)
p_load(tidyverse, tidylog, purrr, forcats, visdat, flextable, colorspace, gtsummary, tidymodels, tictoc) #Basic
p_load(textrecipes, tidytext) #Text processing helpers
p_load(parsnip, discrim, naivebayes, ranger, xgboost, kknn, keras, workflowsets, themis, stacks) #Models
doMC::registerDoMC(cores = parallel::detectCores(logical = FALSE)) #Parallel processing
set.seed(100)

#Load data
df <- readRDS("Data/2020 Mortality Data_clean.rds")

### Load Clinspacy
# https://github.com/ML4LHS/clinspacy
# reticulate::conda_remove('clinspacy')
# remotes::install_github('ML4LHS/clinspacy', force = T)
pacman::p_load(clinspacy)

# Initiate clinspacy / use UMLS linker (this may take a very long time)
clinspacy_init(use_linker = TRUE)

# Load cui2vec
cui2vec_definitions <- dataset_cui2vec_definitions()

cui2vec_embed <- dataset_cui2vec_embeddings()
cui2vec_embed <- cui2vec_embed %>% rename(token = cui)
cui2vec_embed <- as_tibble(cui2vec_embed)
saveRDS(cui2vec_embed, "Data/cui2vec_embed.RDS")
cui2vec_embed <- readRDS("cui2vec_embed.RDS")
cui2vec_embed <- as_tibble(cui2vec_embed)

# Parse DF to CUIs
df.cui <- clinspacy(df, df_col = "text", df_id = "id")

# Save
#saveRDS(df.cui, "Data/df_cui_comma.RDS")

# Use only organic chemicals
df.cui.org <- df.cui %>%
  filter(semantic_type == "Organic Chemical")

# Collapse all CUI's to each individual death entry
df.cui.org <- df.cui.org %>%
  group_by(clinspacy_id) %>%
  distinct(entity, .keep_all = T) %>%
  summarize(text = str_c(cui, collapse = " "))

#Merge back to origina DF
df.cui2 <- df %>%
  left_join(df.cui.org %>%
              select(id = clinspacy_id, cui = text),
            by = "id"
  )

#Save
#saveRDS(df.cui2, "Data/df_cui2.RDS")

#Set-up outcomes list for map
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

# Set metrics for classification problem
multi_met <- metric_set(f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv)

#Set lists for models
models.list <- c("Logistic regression", "Naive Bayes",  "Random forest", "XGBoost", "KNN", "MLP", "SVM") 
recipe.model.list <- c("recipe_Logistic.regression", "recipe_Naive.Bayes", "recipe_Random.forest", 
                       "recipe_XGBoost", "recipe_KNN",  "recipe_MLP", "recipe_SVM") 

#Set-up specs for classifiers
log_spec <- logistic_reg() %>% 
  set_engine(engine = "glm") %>% 
  set_mode("classification") # model mode
nb_spec <- naive_Bayes() %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")
rf_spec <- rand_forest(min_n = tune(), trees = tune(), mtry  = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")
xgb_spec <- boost_tree(trees = tune(), tree_depth = tune(), mtry  = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")
knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("classification") 
svm_spec <- svm_linear(cost = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")
nnet_spec <-  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_mode("classification") %>%
  set_engine("nnet")

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
      models = list(Logistic.regression = log_spec, Naive.Bayes = nb_spec, Random.forest = rf_spec, 
                    XGBoost = xgb_spec, KNN = knn_spec, SVM = svm_spec, MLP = nnet_spec))
)

#Set up parameter tunning
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
      grid = 10,
      control = grid_ctrl,
      metrics = metric_set(recall, precision, f_meas, accuracy, kap, roc_auc, sens, spec, ppv, npv),
      )
)
time.training <- toc()


#Save/Load to local harddrive
saveRDS(grid_results.list, paste0("Output/grid_results", as.character(Sys.Date()),"cui_emb.rds"))

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

###Test data
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
saveRDS(test_results, paste0("Output/test_results", as.character(Sys.Date()),"_cui_embed.rds"))

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
openxlsx::write.xlsx(error_analysis_text, paste0("Error/error_analysis_df_", Sys.Date(), "_cui_embed.xlsx"))

error_analysis_all_data <- map(1:length(outcomes), ~ {
  error_analysis[[.x]] %>% left_join(df %>% mutate(.row = row_number()), by = ".row")
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
saveRDS(bootstrap_results, paste0("Output/bootstrap_results", as.character(Sys.Date()),"_cui_embed.rds"))

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

###Export to word}
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
write_word_plot(statistic.plot.list, my_doc1) 
write_word_plot(statistic.best.plot.list, my_doc1) 
write_word_plot(test_CM_table, my_doc1) 

#Create word doc
print(my_doc1, target = paste0("Tables/NLP_tables_cui_embed", Sys.Date(), ".docx")) %>% invisible()
