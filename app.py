from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model and data
with open("model.pkl", "rb") as f:
    df, tfidf, tfidf_matrix = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    recipes = []
    if request.method == "POST":
        user_ingredients = request.form['ingredients']
        allergies = request.form.getlist('allergies')
        user_vec = tfidf.transform([user_ingredients.lower()])
        cosine_sim = cosine_similarity(user_vec, tfidf_matrix)
        top_indices = cosine_sim[0].argsort()[-10:][::-1]
        for idx in top_indices:
            recipe_ingredients = df.iloc[idx]['cleaned-ingredients']
            if not any(allergy.lower() in recipe_ingredients for allergy in allergies):
                recipes.append({
                    'name': df.iloc[idx]['translatedrecipename'],
                    'ingredients': recipe_ingredients
                })
    return render_template("index.html", recipes=recipes)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)