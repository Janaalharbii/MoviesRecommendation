from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
data, cosine_sim = joblib.load('movie_recommender.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    request_data = request.get_json()
    title = request_data['title'].lower()
    indices = pd.Series(data.index, index=data['title'].str.lower()).drop_duplicates()

    # Improve search to include partial matches
    matching_indices = indices[indices.index.str.contains(title)]
    
    if matching_indices.empty:
        return jsonify({'error': 'Movie not found'})

    idx = matching_indices[0]  # Take the first match
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = data[['title', 'overview']].iloc[movie_indices].to_dict('records')

    return jsonify({'recommended_movies': recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)
