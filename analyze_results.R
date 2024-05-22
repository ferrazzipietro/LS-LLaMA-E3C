# Load the ggplot2 library
library(ggplot2)
library(tidyverse)

data <- read.csv("./data/evaluation_table5Epochs.csv") #"/Users/pietroferrazzi/Desktop/dottorato/mistral_finetuning/data/evaluation_results/joint_result.csv" 

library(writexl)

data %>% head(1)
data%>% summary
plot_results <- function(res, col_name, model_type) {
  ggplot(res, aes_string(x = col_name, y = "f1")) +
    # aes(group = col_name)
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(x = col_name, y = "Mean F1 Score", title = paste("Mean F1 Score by", col_name, 'model = ', model_type)) +
    theme_minimal()
}

plot_boxplots <- function(data, col_name, model_type) {
  ggplot(data, aes_string(x = col_name, y = "f1", group=col_name)) +
    geom_boxplot(fill = "skyblue") +
    labs(title = paste("Box plot of F1 Score by", col_name, "  | ", model_type), y = "F1 Score") +
    theme_minimal()
}

plot_boxplots(data, 'r', '')

show_results_grouped_finetuning <- function(data,
                                            f1_minimum_threshold=0){
  cols <- c('r', 'lora_alpha', 'lora_dropout', 'gradient_accumulation_steps', 'learning_rate')
  data <- data %>%
    filter(
      f1 > f1_minimum_threshold
    )
  for (i in 1:length(cols)){
    print(cols[i])
    
    res <- data %>% 
      #filter(model==model_name) %>%
      group_by(# model,
        !!sym(cols[i])
        )
    print(plot_boxplots(res, as.character(cols[i]),
                        'model'))
    
    # 
    # aggregated <- res %>%
    #   summarise(n_obs = n(),
    #     f1=mean(f1_score),
    #     precision=mean(precision),
    #     recall=mean(recall))
    # p <- plot_results(aggregated, cols[i], model_type)
    # print(p)
  }
}

data%>% filter(gradient_accumulation_steps==8) 

ggplot(data, aes_string(x = 'gradient_accumulation_steps', y = "f1", group='gradient_accumulation_steps')) +
  geom_boxplot(fill = "skyblue") +
  labs(title = paste("Box plot of F1 Score by", 'gradient_accumulation_steps', "  | "), y = "F1 Score") +
  theme_minimal()



show_results_grouped_finetuning(data,
                                f1_minimum_threshold=0) 
  
data %>% filter(model_type=='mistral') %>%
  group_by(quantization) %>%
  summarise( f1_top = max(f1)) %>%
  select(quantization, f1_top)%>%
  arrange(desc(f1_top))

data %>% filter(model_type=='llama') %>%
  arrange(desc(f1)) %>% 
  select (quantization, f1, precision, recall) %>% 
  head(5)

data %>% head(3)
data %>% 
  filter(model_type=='llama') %>% 
  select(model_configurations) %>%
  group_by(model_configurations) %>%
  summarize(n())

library(xlsx)

data %>%
  group_by(model_type, model_size, quantization, fine_tuning) %>%
  top_n(1, f1) %>%
  arrange(desc(f1)) %>%
  select( f1, recall, precision, nShotsInference) %>%
  filter(model_type=='mistral',
         #model_size=='13B'
         ) #%>%
  as.data.frame() %>%
  write.xlsx(file='/Users/pietroferrazzi/Desktop/res_tmp_qwen.xlsx')
