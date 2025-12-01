import json
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import argparse

# 从文件加载 y_true
def load_y_true(ground_truth_file):
    y_true = {}
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            y_true[data["id"]] = 1 if data["hallucination"] == "true" else 0  # 转换为 1 或 0
    return y_true

# 从文件加载 y_scores，并进行反转处理
def load_y_scores(hallucination_file):
    y_scores = {}
    with open(hallucination_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            max_rate = float(data["score"])  # 转换为浮点数
            y_scores[data["id"]] = 1 - max_rate  # 反转得分方向
    return y_scores

# 计算 Youden's Index 并返回最优阈值
def get_optimal_threshold(fpr, tpr, thresholds):
    youden_index = tpr - fpr
    optimal_threshold_index = youden_index.argmax()
    optimal_threshold = thresholds[optimal_threshold_index]
    return optimal_threshold

# 计算幻觉率
def calculate_hallucination_rate(y_true):
    hallucination_count = sum(y_true)
    total_samples = len(y_true)
    hallucination_rate = hallucination_count / total_samples
    return hallucination_rate

# 计算假阳率
def calculate_false_positive_rate(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr

def main(ground_truth_file, hallucination_file):
    # 加载数据
    y_true_dict = load_y_true(ground_truth_file)
    y_scores_dict = load_y_scores(hallucination_file)

    # 将数据按 ID 对齐
    y_true = [y_true_dict[key] for key in sorted(y_true_dict.keys())]
    y_scores = [y_scores_dict[key] for key in sorted(y_scores_dict.keys())]

    # 计算 AUC-ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 计算 Pearson’s r
    pearson_r, _ = pearsonr(y_true, y_scores)

    # 计算 Spearman’s ρ
    spearman_rho, _ = spearmanr(y_true, y_scores)

    # 计算 Kendall’s τ
    kendall_tau, _ = kendalltau(y_true, y_scores)

    # 获取最优阈值（基于 Youden's Index）
    optimal_threshold = get_optimal_threshold(fpr, tpr, thresholds)
    print(f"Optimal Threshold (based on Youden's Index): {optimal_threshold:.4f}")

    # 使用最优阈值进行预测
    predictions = [1 if score >= optimal_threshold else 0 for score in y_scores]

    # 计算 Accuracy, Recall 和 F1 Score
    accuracy = accuracy_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    # 计算幻觉率
    hallucination_rate = calculate_hallucination_rate(y_true)

    # 计算假阳率（FPR）
    fpr_optimal = calculate_false_positive_rate(y_true, predictions)

    # 输出结果
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"Pearson's r: {pearson_r:.4f}")
    print(f"Spearman's ρ: {spearman_rho:.4f}")
    print(f"Kendall's τ: {kendall_tau:.4f}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Hallucination Rate: {hallucination_rate:.4f}")
    print(f"False Positive Rate (FPR): {fpr_optimal:.4f}")

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 标记最优阈值点
    optimal_index = (thresholds >= optimal_threshold).argmax()
    plt.scatter(fpr[optimal_index], tpr[optimal_index], color='red', marker='x', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate AUC-ROC, Recall, F1 Score, FPR, and correlation coefficients for hallucination detection.")
    parser.add_argument("--ground_truth_file", type=str, required=True, help="Path to the ground truth file.")
    parser.add_argument("--hallucination_file", type=str, required=True, help="Path to the hallucination file.")
    
    args = parser.parse_args()
    
    main(args.ground_truth_file, args.hallucination_file)
