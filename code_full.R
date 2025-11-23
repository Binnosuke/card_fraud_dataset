# ===================================================================
# MASTER SCRIPT: CREDIT CARD FRAUD DETECTION (FULL VISUALIZATION)
# ===================================================================

# --- 1. LOAD LIBRARIES ---
library(tidyverse) 
library(skimr)
library(e1071)
library(corrplot)
library(caTools)
library(caret)
library(ROSE)
library(smotefamily)
library(pROC)      # Cho DeLong Test và ROC
library(PRROC)     # Cho PR Curve
library(MLmetrics)
library(xgboost)
library(precrec)
library(randomForest)
library(Rborist)   # Random Forest nhanh
library(car)
library(dplyr)

cat("--- [INIT] Đã tải xong thư viện ---\n\n")

# ===================================================================
# PART 1: PREPROCESSING
# ===================================================================
cat("\n=== PART 1: TIỀN XỬ LÝ DỮ LIỆU ===\n")

# 1. Import Data
  if(!file.exists("creditcard.csv")) stop("LỖI: Không tìm thấy file 'creditcard.csv' trong thư mục làm việc!")
  ds.raw <- read.csv("creditcard.csv")
  
  #summary_raw
  head(ds.raw,3)
  summary_raw <- skim(ds.raw)
  #write.csv(summary_raw, "summary_raw.csv", fileEncoding = "UTF-8", row.names = FALSE)

# 2. Cleaning
  dup_count <- sum(duplicated(ds.raw))
  na_count <- sum(is.na(ds.raw))
  
  ds.clean <- ds.raw %>% 
    drop_na() %>% 
    distinct()
  cat(sprintf("Data gốc: %d dòng. Sau khi lọc: %d dòng.\n", nrow(ds.raw), nrow(ds.clean)))
  
  #summary clean dataset
  summary_clean <- skim(ds.clean)
  #write.csv(summary_clean, "summary_clean.csv", fileEncoding = "UTF-8", row.names = FALSE)
  #duplicates <- ds.raw[duplicated(ds.raw) | duplicated(ds.raw, fromLast = TRUE), ]
  #write.csv(duplicates, "duplicates.csv", row.names = FALSE)
  
  #check if the removal of duplicate rows affects the PCA components
  #mean(abs(colMeans(ds.raw[, 2:29]) - colMeans(ds.clean[, 2:29]))) #~1e-3 => good, no need to rescale

# ===================================================================
# PART 2: FULL EDA
# ===================================================================
cat("\n=== PART 2: EDA ===\n")

  # --- PLOT 1: Correlation Matrix (correlation.png) ---
  cat(">> [1/10] Vẽ Correlation Matrix...\n")
  correlations <- cor(ds.clean, method="pearson")
  corrplot(correlations, number.cex = .6, method = "circle", type = "full", tl.cex=0.5, tl.col = "black")

  # CLASS
  summary_class <- ds.clean %>%
    count(Class) %>%
    mutate(percentage = round(n / sum(n) * 100, 4))
  cat(">> Summarise class ratio\n")
  print(summary_class)
  
# NON PCA-TRANSFORMED VARS: Amount & Time
  ##Overall
  print(summary(ds.clean %>% select(Time, Amount)))
  
  # --- PLOT 2: Density Amount Original (density_amount.png) ---
  cat(">> [2/10] Vẽ Density Amount (Gốc)...\n")
    ggplot(ds.clean, aes(x = Amount)) +
      geom_density(fill = "skyblue", alpha = 0.6) +
      labs(title = "Density Plot – Amount (Original Scale)", x = "Amount", y = "Density") +
      theme_minimal()

  # --- PLOT 3: Density Time General (density_time.png) ---
  cat(">> [3/10] Vẽ Density Time (Chung)...\n")
  ggplot(ds.clean, aes(x = Time)) +
    geom_density(fill = "lightgreen", alpha = 0.6) +
    labs(title = "Density Plot – Time", x = "Time", y = "Density") +
    theme_minimal()

  # Feature Engineering - scale
  ds.clean <- ds.clean %>% 
    mutate(
      logAmount = log1p(Amount),
      logAmount_Scaled = as.numeric(scale(logAmount)), 
      Time_Z = as.numeric(scale(Time)),
      Hour = floor(Time / 3600) %% 24,
      Day = paste("Day", floor((Time / 3600) / 24) + 1), 
      Hour_Range = cut(Hour, breaks = 0:24, right = FALSE, labels = paste0(0:23, "-", 1:24)),
      Class = factor(Class, levels = c(0, 1), labels = c("NonFraud", "Fraud"))
    )

  # --- PLOT 4: Density LogAmount by Class (density_logamount.png) ---
  cat(">> [4/10] Vẽ Phân phối Log-Amount theo Class...\n")
  ggplot(ds.clean, aes(x = logAmount_Scaled, fill = Class)) +
    geom_density(alpha = 0.3) +
    scale_fill_manual(values = c("#F8766D", "#00BFC4")) +
    labs(title = "Phân phối Log-Amount (Chuẩn hóa)", x = "Log(Amount + 1)")

  # --- PLOT 5: Distribution of Transaction Time (density_time_class.png) ---
  cat(">> [5/10] Vẽ Distribution of Transaction Time (Density theo Class)...\n")
  ggplot(ds.clean, aes(x = Time, fill = Class)) +
    geom_density(alpha = 0.3) +
    scale_fill_manual(values = c("#F8766D", "#00BFC4"), labels = c("Non-Fraud", "Fraud")) + 
    labs(title = "Distribution of Transaction Time", x = "Time", y = "density") +
    theme_minimal()

  # --- PLOT 6: Facet Bar Chart Time/Day (time_numcount.png) ---
  cat(">> [6/10] Vẽ Số giao dịch theo ngày, giờ (Facet Grid)...\n")
  ggplot(ds.clean, aes(x = Hour_Range, fill = Class)) +
    geom_bar(alpha = 0.6, position = "identity") +
    facet_wrap(Day ~ Class, scales = "free_y", ncol = 2) +
    scale_fill_manual(values = c("#F8766D", "#00BFC4")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7)) +
    labs(title = "Số giao dịch theo ngày, giờ tính từ giao dịch đầu tiên",
         x = "Hour Range",
         y = "Count",
         fill = "Class")

# PCA-TRANSFORMED VARS: V1:V28
  # --- CHUẨN BỊ DATA CHO BOXPLOTS ---
  ds.pca <- ds.clean %>% 
    select(Class, V1:V28) %>% 
    pivot_longer(cols = V1:V28, names_to = "Feature", values_to = "Value") %>%
    mutate(Feature = factor(Feature, levels = paste0("V", 1:28)))

  # --- PLOT 7: Boxplots All Variables (pca_boxplots_notzoom.png) ---
  cat(">> [7/10] Vẽ Boxplots toàn bộ biến (Không Zoom)...\n")
  ggplot(ds.pca, aes(x = factor(Class), y = Value, fill = factor(Class))) +
    geom_boxplot(alpha = 0.7) +
    facet_wrap(~Feature, scales = "free_y", ncol = 7) +
    theme_minimal() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    scale_fill_manual(values = c("skyblue", "salmon"), labels = c("Non-Fraud", "Fraud")) +
    labs(title = "PCA Components - Boxplots",
         x = "Class",
         y = "Value")
  
  # CHECK OUTLIERS
  # 1. Khởi tạo cột đếm
  ds.outlier.counts <- ds.clean %>%
    mutate(num_outliers = 0)
  
  # 2. Chạy vòng lặp check từng biến V1 -> V28
  for (col in paste0("V", 1:28)) {
    Q1 <- quantile(ds.outlier.counts[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(ds.outlier.counts[[col]], 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR_val
    upper_bound <- Q3 + 1.5 * IQR_val
    
    ds.outlier.counts <- ds.outlier.counts %>%
      mutate(num_outliers = num_outliers + (!!sym(col) < lower_bound | !!sym(col) > upper_bound))
  }
  
  # 3. Tính tổng số dòng bị ảnh hưởng
  total_rows <- nrow(ds.clean)
  rows_with_outliers <- sum(ds.outlier.counts$num_outliers > 0)
  pct_affected <- (rows_with_outliers / total_rows) * 100
  
  cat(sprintf("\n=== KẾT QUẢ PHÂN TÍCH THEO HÀNG (ROW-WISE) ===\n"))
  cat(sprintf("Tổng số giao dịch: %d\n", total_rows))
  cat(sprintf("Số giao dịch có ít nhất 1 biến ngoại lai: %d (Chiếm %.2f%%)\n", rows_with_outliers, pct_affected))
  
  
  # 4. Xem chi tiết phân phối (từ 0 đến 28)
  cat("\n--- Phân phối số lượng Outlier trên mỗi giao dịch (Full Range 0-28) ---\n")
  
  full_sequence <- tibble(num_outliers = 0:28)
  actual_counts <- ds.outlier.counts %>% count(num_outliers)
  
  table_outliers <- full_sequence %>%
    left_join(actual_counts, by = "num_outliers") %>%
    mutate(n = coalesce(n, 0)) %>%  # Thay NA bằng 0
    mutate(percentage = round(n / sum(n) * 100, 8))
  print(table_outliers, n = 29)
  
  # --- PLOT 8: Outliers Count (outliers_count.png) ---
  plot_outliers <- ds.outlier.counts %>%
    count(Class, num_outliers) %>%
    complete(num_outliers = 0:28, Class, fill = list(n = 0))
  
  cat(">> [8/10] Vẽ Column Chart thống kê số giao dịch có ở mỗi mốc Outliers_count...\n")
  ggplot(plot_outliers, aes(x = factor(num_outliers), y = n, fill = Class)) +
    geom_col(position = "dodge", width = 0.8) + 
    geom_text(aes(label = ifelse(n > 0, n, 0)), 
              position = position_dodge(width = 0.8),
              vjust = -0.5,
              size = 2.5) +
    scale_y_continuous(trans = "log1p", breaks = c(0, 1, 10, 100, 1000, 10000, 100000)) +
    scale_x_discrete(limits = as.character(0:28)) +
    scale_fill_manual(values = c("skyblue", "salmon"), labels = c("Non-Fraud", "Fraud")) +
    labs(title = "Distribution of Outliers per Transaction",
         subtitle = paste0("Numbers on top represent count of transactions. Log1p Scale used (0 counts are shown at baseline)"),
         x = "Number of Outliers in a Row",
         y = "Count (Log1p Scale)") +
    theme_minimal()
  
  # Thêm feature outliers
  ds.clean$num_outliers <- ds.outlier.counts$num_outliers
  cat(">> [Info] Đã thêm feature 'num_outliers' vào ds.clean.\n")
  
  # --- PLOT 9: Distribution of outliers (density_outliers.png) ---
  cat(">> [9/10] Vẽ Density Plot để so sánh cấu trúc phân phối Fraud...\n")
  plot_data_prop <- ds.clean %>%
    count(Class, num_outliers) %>%
    complete(num_outliers = 0:28, Class, fill = list(n = 0)) %>%
    group_by(Class) %>%
    mutate(pct = n / sum(n)) %>% 
    ungroup()
  
  ggplot(plot_data_prop, aes(x = num_outliers, y = pct, fill = Class, color = Class)) +
    geom_area(alpha = 0.4, position = "identity") + 
    geom_point(size = 1.5, alpha = 0.8) +
    scale_fill_manual(values = c("skyblue", "salmon"), labels = c("Non-Fraud", "Fraud")) +
    scale_color_manual(values = c("dodgerblue4", "firebrick"), labels = c("Non-Fraud", "Fraud")) +
    scale_x_continuous(breaks = seq(0, 28, 2)) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "Distribution of Outliers (Normalized by Class)",
         subtitle = "Non-Fraud tập trung ở 0. Fraud trải rộng (Fat Tail).",
         x = "Number of Outliers (num_outliers)",
         y = "Percentage within Class") +
    theme_minimal()
  
  
  # --- PLOT 9: Boxplots Zoomed (pca_boxplots_zoom.png) ---
  cat(">> [9/9] Vẽ Boxplots Zoom (Đã zoom)...\n")
  # Tính toán thống kê để zoom tự động
  pca.stats <- ds.pca %>%
    group_by(Feature, Class) %>%
    summarise(
      y25 = quantile(Value, 0.25),
      y50 = median(Value),
      y75 = quantile(Value, 0.75),
      IQR = y75 - y25,
      lower_whisker = max(min(Value), y25 - 1.5 * IQR),
      upper_whisker = min(max(Value), y75 + 1.5 * IQR),
      .groups = 'drop'
    )
  
  pca.zoom_limits <- pca.stats %>%
    group_by(Feature) %>%
    summarise(
      view_min = min(y25 - 3 * IQR), 
      view_max = max(y75 + 3 * IQR)
    )
  
  pca.visible_points <- ds.pca %>%
    inner_join(pca.stats, by = c("Feature", "Class")) %>%
    inner_join(pca.zoom_limits, by = "Feature") %>%
    filter((Value < lower_whisker | Value > upper_whisker) & (Value >= view_min & Value <= view_max))
  
  ggplot() +
    geom_boxplot(data = pca.stats, aes(x = Class, fill = Class, ymin = lower_whisker, lower = y25, middle = y50, upper = y75, ymax = upper_whisker), stat = "identity", alpha = 0.7) +
    geom_point(data = pca.visible_points, aes(x = Class, y = Value), size = 1, alpha = 0.5) +
    facet_wrap(~Feature, scales = "free_y", ncol = 7) +
    scale_fill_manual(values = c("skyblue", "salmon"), labels = c("Non-Fraud", "Fraud")) +
    labs(title = "PCA Components - Boxplots (Zoom)", x = "Class", y = "Value") +
    theme_minimal() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

# ===================================================================
# PART 3: HYPOTHESIS TESTS
# ===================================================================
  
  # -------------------------------------------------------------------
  # GT1: KIỂM ĐỊNH SỰ KHÁC BIỆT CỦA AMOUNT
  # -------------------------------------------------------------------
  cat("\n--- GT1: Kiểm định sự khác biệt logAmount giữa Fraud vs Non-Fraud ---\n")
  # H0: Không có sự khác biệt về logAmount giữa 2 nhóm
  # H1: Có sự khác biệt đáng kể
  
  cat(">> Do logAmount_Scaled không tuân theo pp chuẩn -> Sử dụng MANN-WHITNEY U TEST (Non-parametric)\n")
  gt1.test_res <- wilcox.test(logAmount_Scaled ~ Class, data = ds.clean, exact=FALSE)
  
  cat(sprintf(">> Kết quả P-value: %.4e\n", gt1.test_res$p.value))
  if(gt1.test_res$p.value < 0.05) {
    cat(">> Kết luận: BÁC BỎ H0.\n")
    cat("   (Có sự khác biệt có ý nghĩa thống kê về Amount giữa 2 nhóm,\n")
    cat("    tuy nhiên cần kết hợp biểu đồ để xem mức độ tách biệt thực tế.)\n")
  } else {
    cat(">> Kết luận: CHẤP NHẬN H0.\n")
    cat("   (Không tìm thấy bằng chứng về sự khác biệt giữa 2 nhóm.)\n")
  }
  
  # -------------------------------------------------------------------
  # GT2: KIỂM TRA ẢNH HƯỞNG CỦA CÁC BIẾN PCA LÊN VIỆC PHÂN LOẠI (KS-TEST)
  # -------------------------------------------------------------------
  cat("\n--- GT2: Đánh giá mức độ phân loại của từng biến V1-V28 ---\n")
  # H0: Phân phối của biến Vi giữa 2 nhóm là GIỐNG NHAU
  # H1: Phân phối của biến Vi giữa 2 nhóm là KHÁC NHAU
  # Phương pháp: Kolmogorov-Smirnov Test (KS-Test)
  
  gt2.var_ranking <- data.frame(Variable = character(), 
                                KS_Stat = numeric(), 
                                P_Value = numeric(), 
                                stringsAsFactors = FALSE)
  
  for(v in paste0("V", 1:28)){
    gt2.res <- ks.test(ds.clean[[v]][ds.clean$Class=="Fraud"], 
                       ds.clean[[v]][ds.clean$Class=="NonFraud"])
    gt2.var_ranking <- rbind(gt2.var_ranking, 
                             list(Variable = v, 
                                  KS_Stat = gt2.res$statistic,
                                  P_Value = gt2.res$p.value)) }
  
  gt2.var_ranking <- gt2.var_ranking %>% arrange(desc(KS_Stat))
  # In ra các biến tốt và tệ nhất
  cat(">> Tóm tắt kết quả KS-Test: Top 10 biến tốt nhất:\n")
  head(gt2.var_ranking, 10)
  cat(">> Tóm tắt kết quả KS-Test: Top 5 biến tệ nhất:\n")
  tail(gt2.var_ranking, 5)

  # Vẽ Bar Chart để thấy độ dốc của các biến
  ggplot(gt2.var_ranking, aes(x = reorder(Variable, KS_Stat), y = KS_Stat)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    coord_flip() + # Xoay ngang cho dễ đọc tên biến
    geom_hline(yintercept = 0.1, linetype = "dashed", color = "darkblue", linewidth = 0.5) +
    labs(title = "Mức độ tách biệt (KS Statistic) của các biến V",
         subtitle = "Đường đỏ: Ngưỡng khuyến nghị 0.1",
         x = "Biến", y = "KS Statistic (D)") +
    theme_minimal()
  
  good_vars_list <- gt2.var_ranking %>%
    filter(KS_Stat > 0.1) %>%
    pull(Variable)
  
  cat(sprintf(">> Số lượng biến V được giữ lại: %d biến (trên tổng số 28)\n", length(good_vars_list)))
  cat(">> Các biến bị loại bỏ:", setdiff(paste0("V", 1:28), good_vars_list), "\n")
  print(good_vars_list)
  
  cat("\n--- KẾT THÚC PART 3 ---\n")

# ===================================================================
# PART 4: DATA PREP & SAMPLING
# ===================================================================
cat("\n=== PART 4: CHUẨN BỊ DATA & SAMPLING ===\n")

# Prepare ds.model & split dataset
  ds.model <- ds.clean %>% select(num_range("V", 1:28), logAmount_Scaled, num_outliers, Class) 

  set.seed(123)
  idx <- sample.split(Y = ds.model$Class, SplitRatio = 0.7)
  ds.train <- ds.model[idx, ]
  ds.test <- ds.model[!idx, ]
  x_train <- ds.train %>% select(-Class)
  y_train <- ds.train$Class

# SAMPLING
  cat("--- Thực hiện Sampling (Under, Over, SMOTE, ROSE) ---\n")
  
  # 1. Random Undersampling
  set.seed(42)
  ds.train_ru <- downSample(x = x_train, y = y_train, yname = "Class")
  cat("\n1. Dữ liệu train sau Random Undersampling:\n")
  print(table(ds.train_ru$Class))
  
  # 2. Random Oversampling
  set.seed(42)
  ds.train_ro <- upSample(x = x_train, y = y_train, yname = "Class")
  cat("\n2. Dữ liệu train sau Random Oversampling:\n")
  print(table(ds.train_ro$Class))
  
  # 3. SMOTE
  set.seed(42)
  y_train_numeric <- ifelse(y_train == "Fraud", 1, 0)
  smote_data <- smotefamily::SMOTE(X = x_train, target = y_train_numeric, K = 5)
  ds.train_smote <- as.data.frame(smote_data$data)
  colnames(ds.train_smote)[ncol(ds.train_smote)] <- "Class"
  ds.train_smote$Class <- factor(ds.train_smote$Class, 
                                 levels = c(0, 1), 
                                 labels = c("NonFraud", "Fraud"))
  cat("\n3. Dữ liệu train sau SMOTE:\n")
  print(table(ds.train_smote$Class))
  
  # 4. ROSE
  set.seed(42)
  ds.train_rose <- ROSE(Class ~ ., data = ds.train)$data
  cat("\n4. Dữ liệu sau ROSE:\n")
  print(table(ds.train_rose$Class))
  
# QUICK TEST để chọn phương pháp tốt nhất
  # Gom biến
  ds.samp <- list(
    "1_Under" = ds.train_ru, 
    "2_Over"  = ds.train_ro, 
    "3_SMOTE" = ds.train_smote, 
    "4_ROSE"  = ds.train_rose
  )
  
  # Tạo biến lưu kết quả
  samp.results <- list()
  samp.pred_scores <- list()
  class_numeric <- ifelse(ds.test$Class == "Fraud", 1, 0)
  
  cat("\n--- BẮT ĐẦU SO SÁNH CÁC PHƯƠNG PHÁP SAMPLING (Dùng Logistic Regression) ---\n")
  
  for (method_name in names(ds.samp)) {
    cat(paste(">> Đang huấn luyện trên:", method_name, "...\n"))
    current_ds.train <- ds.samp[[method_name]]
    
    set.seed(123)
    model <- glm(Class ~ ., data = current_ds.train, family = "binomial")
    samp.preds <- predict(model, newdata = ds.test, type = "response")
    
    samp.mm <- mmdata(scores = samp.preds, labels = class_numeric)
    samp.curves <- evalmod(samp.mm)
    samp.aucs <- auc(samp.curves) 
    
    # Trích xuất giá trị
    samp.auprc_val <- subset(samp.aucs, curvetypes == "PRC")$aucs
    samp.auroc_val <- subset(samp.aucs, curvetypes == "ROC")$aucs
    
    # Lưu kết quả
    samp.results[[method_name]] <- data.frame(
      Method = method_name,
      AUPRC = samp.auprc_val,
      AUROC = samp.auroc_val
    )
    samp.pred_scores[[method_name]] <- samp.preds
  }
  
  # Tổng hợp bảng kết quả
  samp.results <- do.call(rbind, samp.results) %>%
    arrange(desc(AUPRC))
  
  cat("\n--- BẢNG KẾT QUẢ SO SÁNH ---\n")
  print(samp.results)
  
  best_method <- samp.results$Method[1]
  cat(paste("\n=> Phương pháp sampling tốt nhất (theo AUPRC):", best_method, "\n"))
  train_best <- ds.samp[[best_method]]
  
# VẼ BIỂU ĐỒ PR CHO KẾT QUẢ TRÊN
  samp.mm_all <- mmdata(scores = samp.pred_scores, 
                         labels = rep(list(class_numeric), 4), 
                         modnames = names(samp.pred_scores))
  samp.curves_all <- evalmod(samp.mm_all)
  samp.plot_data <- as.data.frame(samp.curves_all) %>% filter(type == "PRC")
  samp.auc_data <- auc(samp.curves_all) %>%
    filter(curvetypes == "PRC") %>%
    arrange(desc(aucs))
  
  # Tạo legend
  legend_labels <- paste0(samp.auc_data$modnames, " (AUC: ", round(samp.auc_data$aucs, 4), ")")
  names(legend_labels) <- samp.auc_data$modnames
  
  # plot
  ggplot(samp.plot_data, aes(x = x, y = y, color = modname)) +
    geom_line(linewidth = 0.7) +
    geom_hline(yintercept = sum(class_numeric)/length(class_numeric), 
               linetype = "dashed", color = "gray") +
    scale_color_discrete(labels = legend_labels, breaks = samp.auc_data$modnames) +

    labs(title = "So sánh PR Curve giữa các phương pháp Sampling",
         subtitle = "Model: Logistic Regression | Test Set: Original",
         x = "Recall",
         y = "Precision",
         color = "Method & AUPRC") +
    
    theme_minimal() +
    theme(
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 10),
      legend.text = element_text(size = 9),
      legend.background = element_rect(fill = "white", color = "gray90"),
      plot.title = element_text(face = "bold", size = 14)
    )
  
# ===================================================================
# PART 5: MODEL TRAINING
# ===================================================================
cat("\n=== PART 5: TRAINING MODELS ===\n")

# Custom summary cho PR-AUC trong Caret
  custom_summary <- function (data, lev = NULL, model = NULL) {
    if (is.null(lev)) lev <- levels(data$obs)
    probs_pos <- data[, lev[2]]
    labels <- ifelse(data$obs == lev[2], 1, 0)
    pr <- pr.curve(scores.class0 = probs_pos, weights.class0 = labels, curve = FALSE)
    base_metrics <- defaultSummary(data, lev, model)
    c(AUPRC = pr$auc.integral, base_metrics) 
  }
  ctrl <- trainControl(method = "cv", number = 5, 
                       summaryFunction = custom_summary, 
                       classProbs = TRUE, verboseIter = TRUE)

cat("1. Training Logistic Regression...\n")
set.seed(123)
model_lr <- train(Class ~ ., data = train_best, method = "glm", family = "binomial", metric = "AUPRC", trControl = ctrl)

cat("2. Training Random Forest (Rborist)...\n")
set.seed(123)
rf_grid <- expand.grid(predFixed = c(5,6), minNode = c(1,2))
model_rf <- train(Class ~ ., data = train_best, method = "Rborist", metric = "AUPRC", trControl = ctrl, tuneGrid = rf_grid, nTree = 500)

cat("3. Training XGBoost...\n")
set.seed(123)
xgb_grid <- expand.grid(nrounds = c(300,500), max_depth = c(5,7), eta = c(0.05,0.1,0.15), gamma = c(0,0.1), colsample_bytree = c(0.7,0.8), min_child_weight = 1, subsample = 0.8)
model_xgb <- train(Class ~ ., data = train_best, method = "xgbTree", metric = "AUPRC", trControl = ctrl, tuneGrid = xgb_grid, verbosity = 0)

#results
cat("\n--- Kết quả CV (PRAUC) ---\n")
model_results <- resamples(list(Logistic = model_lr, RF = model_rf, XGB = model_xgb))
print(summary(model_results))

# ===================================================================
# PART 6: EVALUATION
# ===================================================================
cat("\n=== PART 6: ĐÁNH GIÁ & KẾT LUẬN ===\n")

eval_full <- function(model, test_df, name) {
  probs <- predict(model, newdata = test_df, type = "prob")$Fraud
  preds <- predict(model, newdata = test_df)
  cm <- confusionMatrix(preds, test_df$Class, positive = "Fraud", mode = "prec_recall")
  mm <- mmdata(scores = probs, labels = class_numeric)
  curve_data <- evalmod(mm)
  aucs <- auc(curve_data)
  auprc_val <- subset(aucs, curvetypes == "PRC")$aucs
  
  # trả về Dataframe 1 dòng
  return(data.frame(
    Model = name,
    AUPRC = auprc_val,
    Recall = cm$byClass["Recall"],
    Precision = cm$byClass["Precision"],
    F1_Score = cm$byClass["F1"],
    Accuracy = cm$overall["Accuracy"]
  ))
}

# chạy đánh giá
res_lr <- eval_full(model_lr, ds.test, "Logistic Regression")
res_rf <- eval_full(model_rf, ds.test, "Random Forest")
res_xgb <- eval_full(model_xgb, ds.test, "XGBoost")

# gộp bảng kết quả
final_tab <- rbind(as.data.frame(res_lr[1:6]), as.data.frame(res_rf[1:6]), as.data.frame(res_xgb[1:6]))
rownames(final_tab) <- NULL
cat("\n--- BẢNG TỔNG HỢP HIỆU SUẤT ---\n")
print(final_tab)


# Vẽ PR Curve Comparison
cat(">> Vẽ biểu đồ PR Curve...\n")
scores_list <- list(
  LR  = predict(model_lr,  newdata = ds.test, type = "prob")$Fraud,
  RF  = predict(model_rf,  newdata = ds.test, type = "prob")$Fraud,
  XGB = predict(model_xgb, newdata = ds.test, type = "prob")$Fraud
)

mm_all <- mmdata(scores = scores_list, 
                 labels = rep(list(class_numeric), 3), 
                 modnames = names(scores_list))

curves <- evalmod(mm_all)
plot_data <- as.data.frame(curves) %>% filter(type == "PRC")

auc_data <- auc(curves) %>%
  filter(curvetypes == "PRC") %>%
  arrange(desc(aucs))

legend_labels <- paste0(auc_data$modnames, " (AUC: ", round(auc_data$aucs, 4), ")")
names(legend_labels) <- auc_data$modnames

baseline <- sum(class_numeric) / length(class_numeric)

ggplot(plot_data, aes(x = x, y = y, color = modname)) +
  geom_line(linewidth = 0.7) +
  geom_hline(yintercept = baseline, linetype = "dashed", color = "gray40") +
  annotate("text", x = 0.1, y = baseline + 0.05, label = "Baseline", 
           color = "gray40", size = 3, hjust = 0) +
  scale_color_discrete(
    name = "Model Performance",
    labels = legend_labels,
    breaks = auc_data$modnames
  ) +

  labs(title = "Precision-Recall Curve Comparison",
       subtitle = "Test Set Evaluation (Original Imbalance)",
       x = "Recall (Sensitivity)",
       y = "Precision (Positive Predictive Value)") +
  
  theme_minimal() +
  theme(
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 10),
    legend.text = element_text(size = 9),
    legend.background = element_rect(fill = "white", color = "gray90"),
    plot.title = element_text(face = "bold", size = 14)
  )


# Feature Importance
cat(">> Vẽ Feature Importance (XGBoost)...\n")
plot(varImp(model_xgb, scale=FALSE), top=20, main="XGBoost Feature Importance")

cat(">> Vẽ Feature Importance (Random Forest)...\n")
plot(varImp(model_rf, scale=FALSE), top=20, main="Random Forest Feature Importance")

cat(">> Vẽ Feature Importance (Linear Regression)...\n")
plot(varImp(model_lr, scale=FALSE), top=20, main="Linear Regression Feature Importance")

cat("\n=== HOÀN THÀNH CHƯƠNG TRÌNH ===\n")