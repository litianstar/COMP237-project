from data_processing import load_data, preprocess_data, get_features_and_labels
from model_training import train_model, evaluate_model
from predict_comments import predict_comments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import pickle

if __name__ == "__main__":
    # 数据加载与预处理 / Data Loading and Preprocessing
    file_path = './Youtube05-Shakira.csv'
    data = load_data(file_path)
    data = preprocess_data(data)

    # 数据探索 / Data Exploration
    print("========== 数据探索 / Data Exploration ==========")
    print("数据基本信息 / Dataset Information:")
    print(data.info())
    print("\n数据样本 / Dataset Samples:")
    print(data.head())

    # 特征提取（Bag-of-Words 和 TF-IDF） / Feature Extraction (Bag-of-Words and TF-IDF)
    print("\n========== 特征提取 / Feature Extraction ==========")
    print("\n使用词袋模型 / Using Bag-of-Words:")
    X_bow, y_bow, vectorizer_bow = get_features_and_labels(data, use_tfidf=False)
    print(f"词袋模型特征矩阵形状 / Bag-of-Words Matrix Shape: {X_bow.shape}")
    print("前 10 个特征名 / Top 10 Features:")
    print(vectorizer_bow.get_feature_names_out()[:10])

    print("\n使用 TF-IDF / Using TF-IDF:")
    X_tfidf, y_tfidf, vectorizer_tfidf, tfidf_transformer = get_features_and_labels(data, use_tfidf=True)
    print(f"TF-IDF 特征矩阵形状 / TF-IDF Matrix Shape: {X_tfidf.shape}")

    # 数据分割 / Data Splitting
    print("\n========== 数据分割 / Data Splitting ==========")
    X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y_bow, test_size=0.25, random_state=42)
    X_train_tfidf, X_test_tfidf = train_test_split(X_tfidf, test_size=0.25, random_state=42)
    print(f"训练集大小 / Training Set Size: {X_train_bow.shape}, 测试集大小 / Testing Set Size: {X_test_bow.shape}")

    # 模型训练与评估（Bag-of-Words） / Model Training and Evaluation (Bag-of-Words)
    print("\n========== 模型训练与评估 - Bag-of-Words / Model Training and Evaluation - Bag-of-Words ==========")
    model_bow = train_model(X_train_bow, y_train)
    evaluate_model(model_bow, X_train_bow, y_train, X_test_bow, y_test)

    # 模型训练与评估（TF-IDF） / Model Training and Evaluation (TF-IDF)
    print("\n========== 模型训练与评估 - TF-IDF / Model Training and Evaluation - TF-IDF ==========")
    model_tfidf = train_model(X_train_tfidf, y_train)
    evaluate_model(model_tfidf, X_train_tfidf, y_train, X_test_tfidf, y_test)

    # 保存模型和矢量器 / Save Models and Vectorizers
    print("\n========== 模型与矢量器保存 / Saving Models and Vectorizers ==========")
    with open('model_bow.pkl', 'wb') as model_bow_file, open('vectorizer_bow.pkl', 'wb') as vectorizer_bow_file:
        pickle.dump(model_bow, model_bow_file)
        pickle.dump(vectorizer_bow, vectorizer_bow_file)
    with open('model_tfidf.pkl', 'wb') as model_tfidf_file, open('tfidf_transformer.pkl', 'wb') as tfidf_transformer_file:
        pickle.dump(model_tfidf, model_tfidf_file)
        pickle.dump(tfidf_transformer, tfidf_transformer_file)
    print("所有模型和矢量器已保存到文件 / All models and vectorizers have been saved.")

    # 测试新评论 / Testing New Comments
    print("\n========== 测试新评论 / Testing New Comments ==========")
    new_comments = [
        "Great job!",                                # Not Spam
        "Win a free iPhone now!",                   # Spam
        "Inspirational content, keep it up!",       # Not Spam
        "Subscribe to my channel for free gifts!",  # Spam
        "I love this video, amazing quality!",      # Not Spam
        "Limited time offer! Click the link now!",  # Spam
        "This music touches my heart deeply.",      # Not Spam
        "Earn $1000 from home without any effort!", # Spam
        "The cinematography is truly beautiful.",   # Not Spam
        "Claim your prize now by clicking here!"    # Spam
    ]

    # 获取 Bag-of-Words 模型预测 / Predictions using Bag-of-Words model
    predictions_bow, probabilities_bow = predict_comments(new_comments, vectorizer_bow, model_bow)

    # 获取 TF-IDF 模型预测 / Predictions using TF-IDF model
    predictions_tfidf, probabilities_tfidf = predict_comments(new_comments, vectorizer_tfidf, model_tfidf, tfidf_transformer=tfidf_transformer)

    # 构造结果表格 / Construct Results Table
    results = pd.DataFrame({
        "ID": range(1, len(new_comments) + 1),
        "Comment": new_comments,
        "Bag-of-Words Prediction": ["垃圾评论 / Spam" if pred == 1 else "非垃圾评论 / Not Spam" for pred in predictions_bow],
        "BOW Spam Probability (%)": [f"{prob * 100:.2f}%" for prob in probabilities_bow],
        "TF-IDF Prediction": ["垃圾评论 / Spam" if pred == 1 else "非垃圾评论 / Not Spam" for pred in predictions_tfidf],
        "TF-IDF Spam Probability (%)": [f"{prob * 100:.2f}%" for prob in probabilities_tfidf]
    })

    # 输出表格 / Print Table
    print("\n========== 预测结果表格 / Prediction Results Table ==========")
    print(results.to_string(index=False))
# 在表格后添加简要说明 / Add a summary explanation after the table
print("\n========== 结果说明 / Result Explanation ==========")
print("""
- BOW Spam Probability (%): Bag-of-Words 模型对垃圾评论的概率评分。
- TF-IDF Spam Probability (%): TF-IDF 模型对垃圾评论的概率评分。
- 概率范围 / Probability Range:
    - 0%-30%: 高可能为非垃圾评论 / High likelihood of Not Spam.
    - 30%-70%: 不确定，建议人工审核 / Uncertain, manual review recommended.
    - 70%-100%: 高可能为垃圾评论 / High likelihood of Spam.
""")
    
