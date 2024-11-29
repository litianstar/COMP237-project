from data_processing import clean_text

def predict_comments(new_comments, vectorizer, model, tfidf_transformer=None):
    """
    预测新评论的类别 / Predict the categories of new comments
    参数 / Args:
        new_comments (list): 新评论列表 / List of new comments
        vectorizer: 已训练的词袋模型矢量器 / Trained Bag-of-Words vectorizer
        model: 已训练的分类模型 / Trained classification model
        tfidf_transformer: 可选的 TF-IDF 转换器 / Optional TF-IDF transformer
    返回值 / Returns:
        tuple: 预测的类别和垃圾评论概率 / Predicted categories and spam probabilities
    """
    # 清洗新评论 / Clean new comments
    cleaned_comments = [clean_text(comment) for comment in new_comments]
    
    # 转换评论为特征矩阵 / Transform comments into feature matrix
    if tfidf_transformer:
        # 如果使用 TF-IDF / If TF-IDF is used
        vectorized_comments = tfidf_transformer.transform(vectorizer.transform(cleaned_comments))
    else:
        # 使用词袋模型特征 / Use Bag-of-Words features
        vectorized_comments = vectorizer.transform(cleaned_comments)
    
    # 预测类别 / Predict categories
    predictions = model.predict(vectorized_comments)
    # 计算垃圾评论概率 / Calculate spam probabilities
    probabilities = model.predict_proba(vectorized_comments)[:, 1]  # 获取垃圾评论概率
    return predictions, probabilities

if __name__ == "__main__":
    import pickle

    # 加载模型和矢量器 / Load model and vectorizer
    with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)

    # 测试新评论 / Test new comments
    new_comments = [
        "This video is amazing!",
        "Win $1000 now! Click here!",
        "Subscribe for free gifts!"
    ]
    predictions, probabilities = predict_comments(new_comments, vectorizer, model)
    for comment, prediction, probability in zip(new_comments, predictions, probabilities):
        print(f"Comment: {comment} -> {'Spam' if prediction == 1 else 'Not Spam'}, Probability: {probability:.2%}")