import functools
from nba_api.stats.static import players # type: ignore
from nba_api.stats.endpoints import playergamelog # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import streamlit as st # type: ignore

# Get player ID (e.g., Scottie Barnes)
def get_player_id(player_name):
    player_dictionary = players.find_players_by_full_name(player_name)
    if player_dictionary:
        return player_dictionary[0]['id']
    else:
        return None

# Get game log data for the player
def get_player_game_log(player_id, season='2023'):
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    gamelog_df = gamelog.get_data_frames()[0]  
    return gamelog_df

# Create averages for points, assists, and rebounds
def preprocess_player_data(player_gamelog):
    player_gamelog['rolling_points'] = player_gamelog['PTS'].rolling(window=5).mean()
    player_gamelog['rolling_assists'] = player_gamelog['AST'].rolling(window=5).mean()
    player_gamelog['rolling_rebounds'] = player_gamelog['REB'].rolling(window=5).mean()

    
    player_gamelog = player_gamelog.dropna()
    return player_gamelog

# Create a target variable 
def create_target_variable(player_gamelog, over_under_line):
    player_gamelog['over_under'] = (player_gamelog['PTS'] > over_under_line).astype(int)
    return player_gamelog

# Caching model training with functools.lru_cache
@functools.lru_cache(maxsize=10)  
def train_model_cached(player_name, over_under_line, season='2023'):
    player_id = get_player_id(player_name)
    if player_id:
        player_gamelog = get_player_game_log(player_id, season)
        player_gamelog = preprocess_player_data(player_gamelog)
        player_gamelog = create_target_variable(player_gamelog, over_under_line)

        features = ['rolling_points', 'rolling_assists', 'rolling_rebounds']
        X = player_gamelog[features]
        Y = player_gamelog['over_under']

        # This code splits the data into training and test sets 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Training Logistic Regression Model
        model = LogisticRegression()
        model.fit(X_train, Y_train)

        # Evaluate the model on test set
        Y_prediction = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_prediction)
        print(f'Model Accuracy: {accuracy:.2f}')
        
        return model
    else:
        return None

# Prediction Function for over/under
def predict_over_under(player_name, over_under_line, season='2023'):
    # Fetch cached model or train a new one
    model = train_model_cached(player_name, over_under_line, season)
    
    if model:
        player_id = get_player_id(player_name)
        player_gamelog = get_player_game_log(player_id, season)
        player_gamelog = preprocess_player_data(player_gamelog)

        # Preparing the latest data to use for prediction
        latest_game = player_gamelog.tail(1)  # Using the player's most recent game data
        X_new = latest_game[['rolling_points', 'rolling_assists', 'rolling_rebounds']]

        # Predicting probability of hitting the over
        probability = model.predict_proba(X_new)[0][1]
        return f"{player_name} has a {probability * 100:.2f}% chance of scoring over {over_under_line} points."
    else:
        return f"Player {player_name} not found."

# Streamlit app for user input
st.title("NBA Over/Under Bet Predictor")

# Input fields for the player name and over/under line with placeholders
player_name = st.text_input("Enter Player Name", placeholder="e.g., LeBron James")  # Placeholder for player name
over_under_line = st.number_input("Enter Over/Under Line", value=0.0)  # Default value for number input

# Predict button
if st.button("Predict"):
    result = predict_over_under(player_name, over_under_line)
    st.write(result)
