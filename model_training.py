from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd

def train_model(X_train, y_train):
    """训练朴素贝叶斯分类器"""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("\n模型性能总结:")
    print(f"交叉验证平均准确率: {cv_scores.mean()}")

    # 测试集评估
    y_pred = model.predict(X_test)
    print(f"测试集准确率: {accuracy_score(y_test, y_pred)}")
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = pd.read_pickle("data_split.pkl")  # 替换为实际路径
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)