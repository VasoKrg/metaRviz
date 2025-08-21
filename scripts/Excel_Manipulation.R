library(readxl)
library(dplyr)
library(openxlsx2)
library(httr)
library(jsonlite)
library(pdftools)
library(writexl)

count_contingency_tables <- function(column, file_path1, sheet1, file_path2, sheet2) {
  table1 <- read_excel(file_path1, sheet = sheet1)
  table2 <- read_excel(file_path2, sheet = sheet2)
  
  joined_df <- table1 %>%
    inner_join(table2, by = "DOI")
  
  doi_counts <- joined_df %>%
    group_by(DOI) %>%
    summarise(total_rows = n())
  
  print(sum(doi_counts$total_rows))
  print(nrow(doi_counts))
  
  counts <- joined_df %>%
    group_by(.data[[column]]) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count))
  
  return(counts)
}

count_studies <- function(column, file_path1, sheet1) {
  table1 <- read_excel(file_path1, sheet = sheet1)
  
  counts <- table1 %>%
    group_by(.data[[column]]) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(desc(count))
  
  return(counts)
}

#############################################################################################
file_path <- "C:/Users/vaso0/Desktop/Meta_Analysis_Tables/Table1.xlsx"
file_path2 <- "C:/Users/vaso0/Desktop/Meta_Analysis_Tables/Breast Cancer Contigency Tables.xlsx"
table1 <- read_excel(file_path, sheet = "BreastCancerVis")

counts1 = count_contingency_tables("Resolution", file_path, "BreastCancerVis", file_path2, "Malignant vs Benign")
counts2 = count_contingency_tables("Resolution", file_path, "BreastCancerVis", file_path2, "Malignant vs Normal")
print(counts1)
print(counts2)

distinct_counts = count_studies("Resolution", file_path, "BreastCancerVis")
print(distinct_counts)

distinct_counts = count_studies("Methodology Type", file_path, "BreastCancerVis")
print(distinct_counts)

counts1 = count_contingency_tables("Methodology Type", file_path, "BreastCancerVis", file_path2, "Malignant vs Benign")
counts2 = count_contingency_tables("Methodology Type", file_path, "BreastCancerVis", file_path2, "Malignant vs Normal")
print(counts1)
print(counts2)
