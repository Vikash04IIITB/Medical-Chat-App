import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    exit(1)

# Load intents
try:
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)
except FileNotFoundError:
    print("Error: intents.json not found")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON in intents.json")
    exit(1)

# Load neural net model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    data = torch.load("data.pth", map_location=device)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
except FileNotFoundError:
    print("Error: data.pth not found")
    exit(1)
except Exception as e:
    print(f"Error loading neural net model: {e}")
    exit(1)

def chat():
    bot_name = "Aarogya"
    print("Let's chat! (type 'quit' to exit)")
    while True:
        try:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                print(f"{bot_name}: Goodbye!")
                break

            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)
            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
            else:
                print(f"{bot_name}: I do not understand...")
        except Exception as e:
            print(f"{bot_name}: Error processing input: {e}")

if __name__ == "__main__":
    chat()