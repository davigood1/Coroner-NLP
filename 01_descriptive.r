# install.packages("pacman")
library(pacman)
p_load(tidyverse, tidylog, purrr, colorspace, flextable) # Basic
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

# Co-occurence table
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
