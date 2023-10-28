import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

data = pd.read_csv('customs_data.csv')
# 处理缺失值
data = data.dropna()

X = data['desc']  # 输入文本数据
y = data['category']  # 分类标签

# X = data.drop('Category', axis=1)  # 特征
# y = data['Category']  # 分类标签

# 示例数据集，包括产品说明文字和它们的分类
# product_descriptions = ["这是一款高清电视", "这是一本小说书", "这是一台洗衣机", "这是一个冰箱", "电灯", "空气净化器", "电吹风", "洗碗机"]
# product_categories = ["电视", "书籍", "家电", "家电", "家电", "家电", "家电", "家电"]

# 文本预处理
nltk.download('stopwords')
stop_words = stopwords.words('chinese')
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将文本数据转换为TF-IDF特征向量
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 训练分类模型
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# 预测产品分类
y_pred = classifier.predict(X_test_tfidf)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("模型准确性：", accuracy)
