from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load("models/movie_rating_model.pkl")
encoder = joblib.load("models/encoder.pkl")  # Dictionary of encoders

@app.route('/')
def index():
    # Get available actors for dropdown
    actor_list = list(encoder['actors'].classes_)
    genre_list = list(encoder['genre'].classes_)
    director_list = list(encoder['director'].classes_)
    
    return render_template('index.html', actor_list=actor_list, genre_list=genre_list, director_list=director_list)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input values
    genre = request.form['genre'].strip()
    director = request.form['director'].strip()
    actors = request.form['actors'].strip()  # Ensure correct formatting

    # Convert encoder classes to a list for comparison
    genre_classes = list(encoder['genre'].classes_)
    director_classes = list(encoder['director'].classes_)
    actor_classes = list(encoder['actors'].classes_)

    # Debugging: Print user input and available classes
    print(f"User Input -> Genre: {genre}, Director: {director}, Actors: {actors}")
    
    # Validate inputs before encoding
    if genre not in genre_classes:
        return render_template('result.html', rating=f"Error: Invalid genre entered. Available: {genre_classes}")

    if director not in director_classes:
        return render_template('result.html', rating=f"Error: Invalid director entered. Available: {director_classes}")

    if actors not in actor_classes:
        return render_template('result.html', rating=f"Error: Invalid actors entered. Available: {actor_classes}")

    try:
        # Encode inputs using LabelEncoder
        genre_encoded = encoder['genre'].transform([genre])[0]
        director_encoded = encoder['director'].transform([director])[0]
        actors_encoded = encoder['actors'].transform([actors])[0]

        # Print encoded values for debugging
        print(f"Encoded Values -> Genre: {genre_encoded}, Director: {director_encoded}, Actors: {actors_encoded}")

        # Prepare the input for the model
        input_data = np.array([[genre_encoded, director_encoded, actors_encoded]])

        # Predict rating
        prediction = model.predict(input_data)

        # Return predicted rating
        return render_template('result.html', rating=round(prediction[0], 2))

    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('result.html', rating="Error: Unable to process request.")

if __name__ == '__main__':
    print("Genres:", encoder['genre'].classes_)
    print("Directors:", encoder['director'].classes_)
    print("Actors:", encoder['actors'].classes_)
    
    app.run(debug=True)
