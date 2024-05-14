# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
from model_training import predict_next_move, encoder
import random

def player(prev_play, opponent_history=[]):
    # Append the opponent's move to the history if it's a valid move
    valid_moves = ['R', 'P', 'S']
    if prev_play in valid_moves:
        opponent_history.append(prev_play)

    # If the length of the history exceeds n, remove the oldest move
    n = 2
    if len(opponent_history) > n:
        opponent_history.pop(0)

    # If the history contains enough moves, predict the next move
    if len(opponent_history) >= n:
        history = [opponent_history[-n:]]
        guess = predict_next_move(history)
    else:
        # If the history doesn't contain enough moves, make a default move or a random move
        guess = random.choice(valid_moves)

    return guess

# def player(prev_play, opponent_history=[]):
#     opponent_history.append(prev_play)
#
#     guess = "R"
#     if len(opponent_history) > 2:
#         guess = opponent_history[-2]
#
#     return guess
