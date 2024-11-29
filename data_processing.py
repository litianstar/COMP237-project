import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import re
import nltk

nltk.download('stopwords')

def load_data(file_path):
    """加载数据并提取必要列"""
    data = pd.read_csv(file_path)
    return data[['CONTENT', 'CLASS']]

def clean_text(text):
    """清洗文本"""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_data(data):
    """清洗数据并返回处理后的数据"""
    data['CONTENT'] = data['CONTENT'].apply(clean_text)
    return data

def get_features_and_labels(data):
    """生成词袋模型特征和标签"""
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['CONTENT'])
    print(f"词袋模型特征矩阵形状: {X.shape}")
    print("前 10 个特征名:")
    print(vectorizer.get_feature_names_out()[:10])
    
    
    """TF-IDF 转换 """
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)
    print(f"TF-IDF 特征矩阵形状: {X_tfidf.shape}")
    
    y = data['CLASS']
    return X_tfidf, y, vectorizer

if __name__ == "__main__":
    file_path = 'Youtube05-Shakira.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    X, y, vectorizer = get_features_and_labels(data)
    print(f"词袋模型特征矩阵形状: {X.shape}")
