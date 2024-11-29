from data_processing import load_data, preprocess_data, get_features_and_labels
from model_training import train_model, evaluate_model
from predict_comments import predict_comments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

if __name__ == "__main__":
    # 数据加载与预处理 Data Loading and Preprocessing
    file_path = './Youtube05-Shakira.csv'
    data = load_data(file_path)
    data = preprocess_data(data)

    # 数据探索 Data Exploration
    print("========== 数据探索 / Data Exploration ==========")
    print("数据基本信息 / Dataset Information:")
    print(data.info())
    print("\n数据样本 / Dataset Samples:")
    print(data.head())

    # 特征提取 Feature Extraction
    X, y, vectorizer = get_features_and_labels(data)
    print(f"词袋模型特征矩阵形状 / Bag-of-Words Matrix Shape: {X.shape}")
    print("前 10 个特征名 / Top 10 Features:")
    print(vectorizer.get_feature_names_out()[:10])

    # 数据分割 Data Splitting
    print("\n========== 数据分割 / Data Splitting ==========")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"训练集大小 / Training Set Size: {X_train.shape}, 测试集大小 / Testing Set Size: {X_test.shape}")

    # 模型训练 Model Training
    print("\n========== 模型训练与评估 / Model Training and Evaluation ==========")
    model = train_model(X_train, y_train)

    # 测试集预测与评估 Test Set Prediction and Evaluation
    y_pred = model.predict(X_test)
    print("\n测试集结果 / Test Set Results:")
    print(f"测试集准确率 / Test Set Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("混淆矩阵 / Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告 / Classification Report:")
    print(classification_report(y_test, y_pred))

    # 性能汇总 Performance Summary
    print("\n========== 性能汇总 / Performance Summary ==========")
    report = classification_report(y_test, y_pred, output_dict=True)
    non_spam_precision = report['0']['precision']
    non_spam_recall = report['0']['recall']
    spam_precision = report['1']['precision']
    spam_recall = report['1']['recall']
    print(f"非垃圾评论的精确率（Precision for Non-Spam）：{non_spam_precision:.2f}")
    print(f"非垃圾评论的召回率（Recall for Non-Spam）：{non_spam_recall:.2f}")
    print(f"垃圾评论的精确率（Precision for Spam）：{spam_precision:.2f}")
    print(f"垃圾评论的召回率（Recall for Spam）：{spam_recall:.2f}")

    # 保存模型和矢量器 Save Model and Vectorizer
    print("\n========== 模型与矢量器保存 / Saving Model and Vectorizer ==========")
    with open('model.pkl', 'wb') as model_file, open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(model, model_file)
        pickle.dump(vectorizer, vectorizer_file)
    print("模型和矢量器已保存到文件 / Model and Vectorizer have been saved.")

    # 测试新评论 Test New Comments
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
    predictions = predict_comments(new_comments, vectorizer, model)
    for comment, prediction in zip(new_comments, predictions):
        print(f"评论 / Comment: {comment} -> {'垃圾评论 / Spam' if prediction == 1 else '非垃圾评论 / Not Spam'}")

    
