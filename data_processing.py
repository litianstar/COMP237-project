import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import re
import nltk

# 下载 nltk 停用词 / Download nltk stopwords
nltk.download('stopwords')

def load_data(file_path):
    """
    加载数据并提取必要列 / Load data and extract necessary columns
    参数 / Args:
        file_path (str): 数据文件路径 / Path to the data file
    返回值 / Returns:
        pd.DataFrame: 包含必要列的数据框 / DataFrame containing the necessary columns
    """
    data = pd.read_csv(file_path)
    return data[['CONTENT', 'CLASS']]

def clean_text(text):
    """
    清洗文本 / Clean the text
    参数 / Args:
        text (str): 原始文本 / Raw text
    返回值 / Returns:
        str: 处理后的文本 / Cleaned text
    """
    text = re.sub(r'\W', ' ', text)  # 去除非单词字符 / Remove non-word characters
    text = text.lower()  # 转为小写 / Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # 去除多余空格 / Remove extra spaces
    return text

def preprocess_data(data):
    """
    清洗数据并返回处理后的数据 / Clean the data and return the processed DataFrame
    参数 / Args:
        data (pd.DataFrame): 原始数据框 / Raw DataFrame
    返回值 / Returns:
        pd.DataFrame: 清洗后的数据框 / Cleaned DataFrame
    """
    data['CONTENT'] = data['CONTENT'].apply(clean_text)
    return data

def get_features_and_labels(data, use_tfidf=False):
    """
    生成特征和标签 / Generate features and labels
    参数 / Args:
        data (pd.DataFrame): 数据框 / DataFrame
        use_tfidf (bool): 是否使用 TF-IDF / Whether to use TF-IDF
    返回值 / Returns:
        tuple: 特征矩阵、标签、词向量器和可选的 TF-IDF 转换器 / Feature matrix, labels, vectorizer, and optional TF-IDF transformer
    """
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['CONTENT'])
    
    if use_tfidf:
        tfidf_transformer = TfidfTransformer()
        X = tfidf_transformer.fit_transform(X)
        return X, data['CLASS'], vectorizer, tfidf_transformer

    return X, data['CLASS'], vectorizer

if __name__ == "__main__":
    # 测试功能 / Test functionality
    file_path = 'Youtube05-Shakira.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    
    # Bag-of-Words 示例 / Bag-of-Words Example
    X_bow, y_bow, vectorizer_bow = get_features_and_labels(data)
    print(f"词袋模型特征矩阵形状 / Bag-of-Words Matrix Shape: {X_bow.shape}")
    print("前 10 个特征名 / Top 10 Features:")
    print(vectorizer_bow.get_feature_names_out()[:10])
    
    # TF-IDF 示例 / TF-IDF Example
    X_tfidf, y_tfidf, vectorizer_tfidf, tfidf_transformer = get_features_and_labels(data, use_tfidf=True)
    print(f"TF-IDF 特征矩阵形状 / TF-IDF Matrix Shape: {X_tfidf.shape}")

