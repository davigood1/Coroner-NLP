#Load libraries
library(pacman)
p_load(tidyverse, tidylog, purrr, forcats, visdat, flextable, colorspace, gtsummary, tidymodels, tictoc, yardstick) #Basic
p_load(textrecipes, tidytext) #Text processing helpers
p_load(parsnip, discrim, naivebayes, ranger, xgboost, kknn, keras, workflowsets, themis, stacks) #Models
#doMC::registerDoMC(cores = parallel::detectCores(logical = FALSE)) #Parallel processing
set.seed(100)

#Load data
df <- readRDS("Data/2020 Mortality Data_clean.rds")
df <- df %>% mutate(across(c(13:34), ~as.factor(.))) #Make factors
df <- df %>% mutate(across(c(13:34), ~fct_rev(.))) #Reverse level for proper testing results. 

#Keyword matching
##Fentanyl
k.fentanyl <- c("fentanyl", "4-ANPP", "carfentanil", "actyelfentanyl")
df <- df %>% 
  mutate(Fentanyl.k = ifelse(grepl(pattern = paste(k.fentanyl, collapse = "|"), text, ignore.case=T), 1,0))

##Heroin
df <- df %>% 
  mutate(Heroin.k = ifelse(grepl(pattern = "Heroin", text, ignore.case=T), 1, 0))

##Prescription opioids
k.rx.opioids <- c("Hydrocodone", "oxycodone", "hydromorphone", "oxymorphone", "codeine", 
                  "oxycontin", "methadone", "percocet", "buprenorphine", 
                  "meperidine", "morphine", "tapentadol", "tramadol", "naltrexone", "levorphanol")
df <- df %>% 
  mutate(`Prescription.opioids.k` = ifelse(grepl(pattern = paste(k.rx.opioids, collapse = "|"), text, ignore.case=T), 1, 0))

##Any opioids
k.any.opioids <- c("fentanyl", "4-ANPP", "carfentanil", "actyelfentanyl", "heroin",
                  "Hydrocodone", "oxycodone", "hydromorphone", "oxymorphone", "codeine", 
                  "oxycontin", "methadone", "percocet", "buprenorphine", 
                  "meperidine", "morphine", "tapentadol", "tramadol", "naltrexone", "levorphanol")
df <- df %>% 
  mutate(`Any.Opioids.k` = ifelse(grepl(pattern = paste(k.any.opioids , collapse = "|"), text, ignore.case=T), 1, 0))

##Methamphetamine
df <- df %>% 
  mutate(Methamphetamine.k = ifelse(grepl(pattern = "Methamphetamine", text, ignore.case=T), 1, 0))
         
##Cocaine                   
k.cocaine <- c("cocaine", "cocaethylene")
df <- df %>% 
  mutate(Cocaine.k = ifelse(grepl(pattern = paste(k.cocaine, collapse = "|"), text, ignore.case=T), 1, 0))

##Benzodiazepines
k.benzodiazepines <- c("Benzodiazapene", "etizolam", "chlordiazepoxide", "lorazepam", "flubromazolam", 
                       "nordiazepam", "diazepam", "pyrazolam", "clonazepam", "estazolam", 
                       "xanax", "alprazolam", "flualprazolam")
df <- df %>% 
  mutate(`Benzodiazepines.k` = ifelse(grepl(pattern = paste(k.benzodiazepines, collapse = "|"), text, ignore.case=T), 1, 0))


##Alcohol
k.alcohol <- c("Alcohol", "ethanol", "ethanolism")
df <- df %>% 
  mutate(`Alcohol.k` = ifelse(grepl(pattern = paste(k.alcohol , collapse = "|"), text, ignore.case=T), 1, 0))

##Others
k.others <- c("Amphetamine", 
              #Anticonvulsants
              "Carbamazepine", "clobazam", "oxcarbazepine", "diazepam", "ethosuxamide", 
              "phenytoin", "gabapentin", "lacosamide", "levetiracetam", "phenobarbital", 
              "pregabalin", "lamotrigine", "topiramate", "valproate", "valproic acid", "zonisamide",
              #Antidepressants
              "Citalopram", "fluoxetine", "fluvoxamine", "paroxetine", "sertraline", "buproprion", 
              "venlafaxine", "duloxetine", "desvenlafasxine", "levomilnacipran", "imipramine", 
              "desipramine", "nortriptyline", "doxepin", "trimipramine", "amoxapine", "protriptyline", 
              "trazodone", "mirtazapine",
              #Antihistamine
              "Diphenydramine", "cetirizine", "chlorpheniramine", "fexofenadine", "loratadine", "hydroxyzine", "doxylamine", "xylazine", 
              #Anti-psychotics
              "Risperidone", "quetiapine", "olanzapine", "aripiprazole", "clozapine", "haloperidol", "chlorpromazine", "ziprasidone", "paliperidone", "trifluoperazine", "perphenazine", "fluphenazine", "lurasidone", "pimozide",
              #Barbiturates
              "Butalbital", "phenobarbital ", "pentobarbital", "butabarbital", "amobarbital", 
              #Hallucinogens
              "Phencycldine", "LSD", "diethylamide", "ketamine", "PCP", "methylenedioxyamphetamine",
              #MDMA
              "3,4-methylenedioxymethamphetamine", "MDMA", "methylenedioxymethamphetamine", "3,4-methylenedioxymethaphetamine",
              #MDA
              "Methylenedioxyamphetamine", "methylenedioxyamphetamine ", "MDA",
              #Muscle relaxants
              "Cyclobenzaprine", "baclofen", "carisoprodol", "metaxalone", "methocarbamol", "tizanidine", "orphenadrine")
df <- df %>% 
  mutate(`Others.k` = ifelse(grepl(pattern = paste(k.others , collapse = "|"), text, ignore.case=T), 1, 0))

##Create factors
df <- df %>% mutate(across(c(Fentanyl.k:Others.k), ~as.factor(.))) #Make factors
df <- df %>% mutate(across(c(Fentanyl.k:Others.k), ~fct_rev(.))) #Reverse level for proper testing results. 

#Set up lists for map
outcomes <- c("Any Opioids", "Heroin", "Fentanyl", "Prescription.opioids", "Methamphetamine", "Cocaine", 
                   "Benzodiazepines", "Alcohol", "Others")
estimates <- c("Any.Opioids.k", "Heroin.k", "Fentanyl.k", "Prescription.opioids.k", 
               "Methamphetamine.k", "Cocaine.k", "Benzodiazepines.k", "Alcohol.k", "Others.k")

outcomes <- list(Any Opioids,Heroin,Fentanyl,Prescription.opioids,Methamphetamine,Cocaine, 
              Benzodiazepines,Alcohol,Others)
estimates <- list(Any.Opioids.k,Heroin.k,Fentanyl.k,Prescription.opioids.k, 
               Methamphetamine.k,Cocaine.k,Benzodiazepines.k,Alcohol.k,Others.k)


multi_met <- metric_set(f_meas, accuracy, kap, sens, spec, ppv, npv)


map2(outcomes, estimates, ~ {
  df %>% multi_met(truth = data[[.x]], estimate = data[[.y]])
})

class.fentanyl <- df %>% 
  multi_met(truth = "Fentanyl", estimate = "Fentanyl.k") %>% 
  mutate(outcome = "Fentanyl")
class.heroin <- df %>% 
  multi_met(truth = "Heroin", estimate = "Heroin.k") %>% 
  mutate(outcome = "Heroin")
class.opioids <- df %>% 
  multi_met(truth = "Any Opioids", estimate = "Any.Opioids.k") %>% 
  mutate(outcome = "Any opioids")
class.rx.opioids <- df %>% 
  multi_met(truth = "Prescription.opioids", estimate = "Prescription.opioids.k") %>% 
  mutate(outcome = "Prescription opioids")
class.methamphetamine <- df %>% 
  multi_met(truth = "Methamphetamine", estimate = "Methamphetamine.k") %>% 
  mutate(outcome = "Methamphetamine")
class.cocaine <- df %>% 
  multi_met(truth = "Cocaine", estimate = "Cocaine.k") %>% 
  mutate(outcome = "Cocaine")
class.benzo <- df %>% 
  multi_met(truth = "Benzodiazepines", estimate = "Benzodiazepines.k") %>% 
  mutate(outcome = "Benzodiazepines")
class.alcohol <- df %>% 
  multi_met(truth = "Alcohol", estimate = "Alcohol.k") %>%
  mutate(outcome = "Alcohol")
class.others <- df %>% 
  multi_met(truth = "Others", estimate = "Others.k") %>%
  mutate(outcome = "Others")

df.classifications <- bind_rows(class.fentanyl, class.heroin, class.opioids, class.rx.opioids, 
                                class.methamphetamine, class.cocaine, class.benzo, class.alcohol, class.others) %>%
  pivot_wider(
    id_cols = "outcome",
    names_from = ".metric",
    values_from = ".estimate") %>%
  select(Substance = outcome, `F-score` = `f_meas`, Accuracy = `accuracy`, Kappa = `kap`,
         `Sensitivity\n(Recall)` = `sens`, Specificity = `spec`,
         `Positive predictive value\n(Precision)` = `ppv`,
         `Negative predictive value` = `npv`) 

tbl.classifications <- flextable::flextable(df.classifications %>%
                                              mutate(across(where(is.numeric), ~round(.x, 3)))) %>%
  set_caption(caption =  "Table. Diagnostic metrics of keyword matching in entire dataset") %>%
  autofit() %>% theme_zebra(odd_header = "transparent", even_header = "transparent")
tbl.classifications
