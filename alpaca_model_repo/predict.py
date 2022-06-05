import os
from model.model import Alpaca_Prediction

def load_model():
    # TODO: replace the random model to your owns
    model = Alpaca_Prediction
    return model

def run(model) -> dict:
    # You should read image from data folder and return model prediction results
    # 0 represents non-alpaca
    # 1 represents alpaca
    # The return needs to be a dictionary of binaries
    results = {}
    for f in os.listdir("data"):
        results[f] = model(f)
    return results

if __name__ == '__main__':
    print(run(load_model()))
