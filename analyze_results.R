# Load the ggplot2 library
library(ggplot2)
library(tidyverse)

data <- read.csv("/Users/pietroferrazzi/Desktop/dottorato/mistral_finetuning/data/evaluation_results/joint_results.csv") #"/Users/pietroferrazzi/Desktop/dottorato/mistral_finetuning/data/evaluation_results/joint_result.csv" 

library(writexl)

# Assuming your tibble is named "your_tibble" and the file name you want to save is "output.xlsx"
write_xlsx(data, "/Users/pietroferrazzi/Desktop/dottorato/mistral_finetuning/output.xlsx")

data %>% head(1)

plot_results <- function(res, col_name, model_type) {
  ggplot(res, aes_string(x = col_name, y = "f1")) +
    # aes(group = col_name)
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(x = col_name, y = "Mean F1 Score", title = paste("Mean F1 Score by", col_name, 'model = ', model_type)) +
    theme_minimal()
}

plot_boxplots <- function(data, col_name, model_type) {
  ggplot(data, aes_string(x = col_name, y = "f1_score", group=col_name)) +
    geom_boxplot(fill = "skyblue") +
    labs(title = paste("Box plot of F1 Score by", col_name, "  | ", model_type), y = "F1 Score") +
    theme_minimal()
}

show_results_grouped_finetuning <- function(data,
                                            f1_minimum_threshold=0){
  cols <- c('maxNewTokensFactor', 'nShotsInference', 'quantization', 'r', 'lora_alpha', 'lora_dropout', 'gradient_accumulation_steps', 'learning_rate')
  data <- data %>% 
    filter(fine_tuning == 'FT',
      !is.na(data['f1_score']),
      f1_score > f1_minimum_threshold
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

show_results_grouped_finetuning(data %>% filter(model_type=='mistral'),
                                f1_minimum_threshold=0.3) 
  
data %>% filter(model_type=='mistral') %>%
  group_by(quantization) %>%
  summarise( f1_top = max(f1_score)) %>%
  select(quantization, f1_top)%>%
  arrange(desc(f1_top))

data %>% filter(model_type=='llama') %>%
  arrange(desc(f1_score)) %>% 
  select (quantization, f1_score, precision, recall) %>% 
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
  top_n(1, f1_score) %>%
  arrange(desc(f1_score)) %>%
  select(model_type, model_size, quantization, fine_tuning, f1_score, recall, precision, nShotsInference) %>%
  filter(model_type=='mistral',
         #model_size=='13B'
         ) #%>%
  as.data.frame() %>%
  write.xlsx(file='/Users/pietroferrazzi/Desktop/res_tmp_qwen.xlsx')
