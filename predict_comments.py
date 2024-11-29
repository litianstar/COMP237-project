from data_processing import clean_text

def predict_comments(new_comments, vectorizer, model):
    """预测新评论的类别"""
    cleaned_comments = [clean_text(comment) for comment in new_comments]
    vectorized_comments = vectorizer.transform(cleaned_comments)
    predictions = model.predict(vectorized_comments)
    return predictions

if __name__ == "__main__":
    import pickle

    # 加载模型和矢量器
    with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)

    # 测试新评论
    new_comments = [
        "This video is amazing!",
        "Win $1000 now! Click here!",
        "Subscribe for free gifts!"
    ]
    predictions = predict_comments(new_comments, vectorizer, model)
    for comment, prediction in zip(new_comments, predictions):
        print(f"Comment: {comment} -> {'Spam' if prediction == 1 else 'Not Spam'}")