### Load libraries
#install.packages("pacman")
library(pacman)
p_load(tidyverse, tidylog, tidymodels, textrecipes, tidytext, workflowsets, readxl)
p_load(clinspacy)

###Set up
set.seed(100)

#Set-up outcomes list
outcomes.list <- c("Any Opioids", "Heroin", "Fentanyl", "Prescription.opioids",
                   "Methamphetamine", "Cocaine", "Benzodiazepines", "Alcohol", "Others")
outcomes <- outcomes.list

#Import finalized models - take a while
test_results.opi <- readRDS("Output/test_results2022-01-28_CUI_embed_opi.rds")
test_results.other <- readRDS("Output/test_results2022-01-18_CUI_embed_other.rds")

test_results <- append(test_results.opi, test_results.other)

#Extract fitted models to predict
test_fitted <- map(1:length(outcomes), ~
                    test_results[[.x]]$.workflow[[1]]
)

#Load clinspacy
pacman::p_load(clinspacy)

#Initiate clinspacy
clinspacy_init(use_linker = TRUE)


###Prepara data function
#Give dataframe with columns text and id
prepare_data <- function(df.text) {
  
  #Place commas inbetween each token/word
  df.text <- df.text %>% mutate(text = gsub("\\s+", ", ", text))

  #Parse DF to CUIs
  df.cui <- clinspacy(df.text, df_col = 'text', df_id = "id",  threshold = 0.70)

  #Use only organic chemicals
  df.cui.org <- df.cui %>% filter(semantic_type == "Organic Chemical" | entity == "opioid") 
  df.cui.org <- df.cui.org %>% group_by(clinspacy_id) %>% distinct(entity, .keep_all = T) %>% 
  summarize(text = str_c(cui, collapse = " "))
  df.cui2 <- df.text %>% left_join(df.cui.org %>% select(id = clinspacy_id, cui = text), by = "id")
  
  return(df.cui2)
}

###Predict function
#Give prepared data with text column and cui column

#Predict complete
predict_drugs_complete <- function(data){
predict.list <- map2(1:length(outcomes), outcomes, ~ {
  newname = paste(.y)
  newname2 = paste0(.y, "_pred_1")
  newname3 = paste0(.y, "_pred_0")  
  bind_cols(
    predict(test_fitted[[.x]], new_data = data, type = "class"),
    predict(test_fitted[[.x]], new_data = data, type = "prob"))  %>%
    rename(!!newname := .pred_class, !!newname2 := .pred_1, !!newname3 := .pred_0)
}
)

#Create dataframe with predictions
predict <- bind_cols(data %>% select(text, cui), bind_cols(predict.list))
return(predict)
}

#Predict small
predict_drugs_small <- function(data){
predict.list <- map2(1:length(outcomes), outcomes, ~ {
  newname = paste(.y)
  df.predict <- predict(test_fitted[[.x]], new_data = data, typ = "class") %>%
  rename(!!newname := .pred_class)
}
)

#Create dataframe with predictions
predict.small <- bind_cols(data, bind_cols(predict.list))
return(predict.small)
}

#Example run - df needs to have ID and text column
df.test <- data.frame(text = "patient overdose fentanyl gabapentin methadone", id = "1")
df.test1 <- prepare_data(df.test)
predict_drugs_small(df.test1)  





