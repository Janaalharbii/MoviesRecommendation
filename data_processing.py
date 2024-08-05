import pandas as pd

# تحميل البيانات
movies = pd.read_csv('tmdb_5000_movies.csv')

# اختيار الأعمدة المناسبة
movies = movies[['id', 'title', 'overview', 'vote_average', 'release_date']]

# تنظيف البيانات (إزالة الصفوف الفارغة)
movies.dropna(inplace=True)

# حفظ البيانات النظيفة
movies.to_csv('cleaned_movies.csv', index=False)

# عرض بعض السجلات بعد التنظيف
print(movies.head())
