from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import pickle

def train_model(X_train, y_train):
    """
    训练朴素贝叶斯分类器 / Train a Naive Bayes Classifier
    参数 / Args:
        X_train: 训练特征矩阵 / Training feature matrix
        y_train: 训练标签 / Training labels
    返回值 / Returns:
        model: 训练好的模型 / Trained model
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    评估模型性能 / Evaluate the model's performance
    参数 / Args:
        model: 训练好的模型 / Trained model
        X_train, y_train: 训练数据 / Training data
        X_test, y_test: 测试数据 / Testing data
    返回值 / Returns:
        None
    """
    # 交叉验证 / Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("\n========== 模型性能总结 / Model Performance Summary ==========")
    print(f"交叉验证平均准确率 / Cross-Validation Average Accuracy: {cv_scores.mean():.2%}")

    # 测试集评估 / Test Set Evaluation
    y_pred = model.predict(X_test)
    print(f"测试集准确率 / Test Set Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("混淆矩阵 / Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告 / Classification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, filepath):
    """
    保存模型 / Save the trained model
    参数 / Args:
        model: 训练好的模型 / Trained model
        filepath: 保存路径 / File path to save the model
    返回值 / Returns:
        None
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"模型已保存到 / Model saved to: {filepath}")

def load_model(filepath):
    """
    加载模型 / Load a trained model
    参数 / Args:
        filepath: 模型路径 / File path of the saved model
    返回值 / Returns:
        model: 加载的模型 / Loaded model
    """
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    print(f"模型已加载 / Model loaded from: {filepath}")
    return model

if __name__ == "__main__":
    # 测试功能 / Test functionality
    import pandas as pd

    # 模拟数据加载 / Simulate data loading
    data_split_path = "data_split.pkl"  # 替换为实际路径 / Replace with actual path
    X_train, X_test, y_train, y_test = pd.read_pickle(data_split_path)

    # 训练模型 / Train model
    model = train_model(X_train, y_train)

    # 评估模型 / Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # 保存模型 / Save model
    save_model(model, "naive_bayes_model.pkl")

    # 加载模型并验证 / Load model and validate
    loaded_model = load_model("naive_bayes_model.pkl")
    evaluate_model(loaded_model, X_train, y_train, X_test, y_test)