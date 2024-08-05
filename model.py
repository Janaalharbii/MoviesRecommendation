import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# تحميل البيانات النظيفة
data = pd.read_csv('cleaned_movies.csv')

# إعداد المدخلات
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])

# حساب تشابه الكوساين بين الأفلام
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# حفظ النموذج
joblib.dump((data, cosine_sim), 'movie_recommender.pkl')
