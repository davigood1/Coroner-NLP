# install.packages("pacman")
library(pacman)
p_load(tidyverse, tidylog, purrr, colorspace, flextable, gtsummary) # Basic
p_load(textrecipes, tidytext, stopwords, wordcloud) # Text processing helpers

set.seed(100)

df <- read_csv("Data/2020 Mortality Data_clean.csv")

# Tables
## Number of cases per county
df %>%
  group_by(County) %>%
  summarise(n = n()) %>%
  mutate(Percentage = round(n / sum(n) * 100, 1)) %>%
  arrange(desc(n))

##Wrangle race/ethnicity
df %>%
  select(Age, Gender...6, Race) %>%
  mutate(Race = case_when(
    str_detect(tolower(Race), "white") ~ "White",
    str_detect(tolower(Race), "black") ~ "Black",
    str_detect(tolower(Race), "latino") ~ "Latino",
    #str_detect(tolower(Race), "LATINO") ~ "Latino",
    str_detect(tolower(Race), "hispanic") ~ "Latino",
    #str_detect(tolower(Race), "HISPANIC") ~ "Latino",
    str_detect(tolower(Race), "asian") ~ "Asian",
    str_detect(tolower(Race), "chinese") ~ "Asian",
    str_detect(tolower(Race), "filipino") ~ "Asian",
    str_detect(tolower(Race), "cambodian") ~ "Asian",
    str_detect(tolower(Race), "indian") ~ "Asian",
    str_detect(tolower(Race), "japanese") ~ "Asian",
    str_detect(tolower(Race), "korean") ~ "Asian",
    str_detect(tolower(Race), "laotian") ~ "Asian",
    str_detect(tolower(Race), "thai") ~ "Asian",
    str_detect(tolower(Race), "vietnamese") ~ "Asian",
    str_detect(tolower(Race), "nat amer") ~ "American Indian or Alaska Native",
    str_detect(tolower(Race), "native american") ~ "American Indian or Alaska Native",
    str_detect(tolower(Race), "am. indian") ~ "American Indian or Alaska Native",
    str_detect(tolower(Race), "am. indian south") ~ "American Indian or Alaska Native",
    str_detect(tolower(Race), "american indian") ~ "American Indian or Alaska Native",
    str_detect(tolower(Race), "armenian") ~ "Middle Eastern",
    str_detect(tolower(Race), "middle eastern") ~ "Middle Eastern",
    str_detect(tolower(Race), "guamanian") ~ "Native Hawaiian and Other Pacific Islander",
    str_detect(tolower(Race), "hawaiian") ~ "Native Hawaiian and Other Pacific Islander",
    str_detect(tolower(Race), "pacific islander") ~ "Native Hawaiian and Other Pacific Islander",
    str_detect(tolower(Race), "samoan") ~ "Native Hawaiian and Other Pacific Islander",
    str_detect(tolower(Race), "multi-racial") ~ "Multiracial",
    str_detect(tolower(Race), "multiracial") ~ "Multiracial",
    str_detect(tolower(Race), "0") ~ "Unknown",
    str_detect(tolower(Race), "null") ~ "Unknown",
    str_detect(tolower(Race), "other") ~ "Unknown",
    str_detect(tolower(Race), "unknown") ~ "Unknown",
    TRUE ~ Race
  )) %>%
  tbl_summary()

## Descriptives for text
df %>%
  summarize(
    Min = min(nchar(text)),
    Q1 = quantile(nchar(text), .25),
    Mean = mean(nchar(text)),
    Median = median(nchar(text)),
    Q3 = quantile(nchar(text), .75),
    Max = max(nchar(text))
  )

df %>%
  mutate(nword = str_count(text, "\\w+")) %>%
  summarize(
    Min = min(nword),
    Q1 = quantile(nword, .25),
    Mean = mean(nword),
    Median = median(nword),
    Q3 = quantile(nword, .75),
    Max = max(nword)
  )



# Frequencies
p.frequency <- map(c("SecondaryIsNA", "County"), ~ {
  df %>%
    ggplot(aes(nchar(text), fill = as.factor(.data[[.x]]))) +
    geom_histogram(binwidth = 1, alpha = 0.8) +
    labs(
      x = "Number of characters per text",
      y = "Number of texts"
    ) +
    theme_minimal()
})

texts <- as_tibble(df) %>%
  mutate(document = row_number()) %>%
  select(text)

tidy_texts <- texts %>%
  unnest_tokens(word, text) %>%
  group_by(word) %>%
  filter(n() > 10) %>%
  ungroup()

# Remove stop words
stopword <- as_tibble(stopwords::stopwords("en"))
stopword <- rename(stopword, word = value)
tb <- anti_join(tidy_texts, stopword, by = "word")

# Frequencies
tb %>%
  count(word, sort = TRUE) %>%
  filter(n > 2000) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_flip() +
  theme_classic(base_size = 12) +
  labs(title = "Word frequency", subtitle = "n > 100") +
  theme(plot.title = element_text(lineheight = .8, face = "bold")) +
  scale_fill_brewer()

# Word cloud
tb %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 15))

# Bigrams
bigrams <- texts %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

bigrams_count <- bigrams %>%
  count(bigram, sort = TRUE)

# seperate words
bigrams_separated <- bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

# filter stop words and NA
bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word) %>%
  filter(!is.na(word1))

# new bigram counts:
bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

# Networks
p_load(igraph, ggraph)

# filter for only relatively common combinations
bigram_graph <- bigram_counts %>%
  filter(n > 1000) %>%
  graph_from_data_frame()

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n),
    show.legend = FALSE,
    arrow = a, end_cap = circle(.07, "inches")
  ) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()

df.long <- df %>% pivot_longer(
  cols = 13:34
)

#Frequency plots
ggplot(
  data = df.long %>% filter(value == 1) %>%
    filter(!name %in% c("Dextromethorphan", "Opioid", "Xanax", "Others", "Flualprazolam")) %>%
    mutate(name = ifelse(name == "Prescription.opioids", "Prescription opioids", name)),
  aes(x = reorder(name, name, function(x) -length(x)))
) +
  geom_bar(stat = "count", position = position_dodge(), fill = "navy blue") +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  scale_y_continuous(limits = c(0, 6500)) +
  labs(x = "Substance", y = "Count") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, size = 14, hjust = 1),
    legend.position = "none"
  )
ggsave("Plots/fig 2.tiff", height = 8, width = 10, dpi = 300)

#Percentage plot - Requested by reviewer
df.perc <- df.long %>% 
  filter(!name %in% c("Dextromethorphan", "Opioid", "Xanax", "Others", "Flualprazolam")) %>%
  group_by(name, value) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n) * 100) %>%
  filter(value == 1) %>%
  filter(!name %in% c("Dextromethorphan", "Opioid", "Xanax", "Others", "Flualprazolam")) %>%
  arrange(desc(freq))
df.perc$name <- ifelse(df.perc$name == "Prescription.opioids", "Prescription opioids", df.perc$name)
df.perc$name <- factor(df.perc$name, levels = unique(df.perc$name[order(desc(df.perc$freq))])) 
 
ggplot(data = df.perc, aes(x = name, y = freq)) +
  geom_bar(stat = "identity", fill = "navy blue", alpha = 0.9) +
  geom_text(aes(label = paste0(round(freq,1),"%")), stat = "identity", vjust = -0.5) +
  ggpubr::geom_bracket(y.position = 10, xmin = "Anticonvulsant", xmax = "MDA", label = 'Grouped as "Others"',
                       label.size = 6, size = 1) +
  labs(x = "Substance", y = "Percentage") +
  scale_y_continuous(limits = c(0,20)) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, size = 14, hjust = 1),
    text = element_text(size = 14),     
    legend.position = "none"
  )
ggsave("Plots/fig 2_percentage.tiff", height = 8, width = 10, dpi = 300)
ggsave("Plots/fig 2_percentage.pdf", height = 8, width = 10, dpi = 300)


# How many are all negative
ggplot(
  data = df %>% group_by(`Number of substances`) %>% count() %>% uncount(n),
  aes(x = as.factor(`Number of substances`), fill = `Number of substances`)
) +
  geom_bar(stat = "count", position = position_dodge()) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  scale_y_continuous(limits = c(0, 27000), breaks = c(0, 5000, 10000, 15000, 20000, 25000)) +
  labs(x = "Number of Substance(s) Classified", y = "Count") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, size = 14, hjust = 0.5),
    legend.position = "none"
  )

# Co-occurence table - Supplement table
df.cross <- as.data.frame(
  crossprod(
    as.matrix(df[, c(
      "Heroin", "Fentanyl", "Prescription.opioids",
      "Methamphetamine", "Cocaine", "Benzodiazepines", "Alcohol", "Others"
    )] == 1)
  )
)

flextable(df.cross %>% rownames_to_column()) %>%
  set_caption(caption = "Supplementary Table. Co-occurrence of substances involved in overdose deaths") %>%
  autofit() %>%
  theme_zebra(odd_header = "transparent", even_header = "transparent")
