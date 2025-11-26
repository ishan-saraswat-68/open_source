import spacy
from spacy.training.example import Example
import random
import os
from pathlib import Path

# Define the output directory
output_dir = Path(__file__).parent / "model"

# Expanded training data with more diverse examples
TRAIN_DATA = [
    # Basic patterns
    ("I had 2 apples", {"entities": [(6, 7, "QUANTITY"), (8, 14, "FOOD")]}),
    ("I ate one banana", {"entities": [(6, 9, "QUANTITY"), (10, 16, "FOOD")]}),
    ("3 cups of rice", {"entities": [(0, 1, "QUANTITY"), (2, 6, "UNIT"), (10, 14, "FOOD")]}),
    ("200g chicken breast", {"entities": [(0, 3, "QUANTITY"), (3, 4, "UNIT"), (5, 19, "FOOD")]}),
    ("a glass of milk", {"entities": [(0, 1, "QUANTITY"), (2, 7, "UNIT"), (11, 15, "FOOD")]}),
    ("two slices of bread", {"entities": [(0, 3, "QUANTITY"), (4, 10, "UNIT"), (14, 19, "FOOD")]}),
    ("500 ml water", {"entities": [(0, 3, "QUANTITY"), (4, 6, "UNIT"), (7, 12, "FOOD")]}),
    ("1 serving of pasta", {"entities": [(0, 1, "QUANTITY"), (2, 9, "UNIT"), (13, 18, "FOOD")]}),
    ("half a pizza", {"entities": [(0, 4, "QUANTITY"), (7, 12, "FOOD")]}),
    ("quarter pound burger", {"entities": [(0, 7, "QUANTITY"), (8, 13, "UNIT"), (14, 20, "FOOD")]}),
    ("100 grams of spinach", {"entities": [(0, 3, "QUANTITY"), (4, 9, "UNIT"), (13, 20, "FOOD")]}),
    ("one large orange", {"entities": [(0, 3, "QUANTITY"), (10, 16, "FOOD")]}),
    ("3 eggs", {"entities": [(0, 1, "QUANTITY"), (2, 6, "FOOD")]}),
    ("a bowl of cereal", {"entities": [(0, 1, "QUANTITY"), (2, 6, "UNIT"), (10, 16, "FOOD")]}),
    ("I consumed 2.5 liters of water", {"entities": [(11, 14, "QUANTITY"), (15, 21, "UNIT"), (25, 30, "FOOD")]}),
    ("ate a sandwich", {"entities": [(4, 5, "QUANTITY"), (6, 14, "FOOD")]}),
    ("had steak for dinner", {"entities": [(4, 9, "FOOD")]}),
    ("10 oz steak", {"entities": [(0, 2, "QUANTITY"), (3, 5, "UNIT"), (6, 11, "FOOD")]}),
    ("5 strawberries", {"entities": [(0, 1, "QUANTITY"), (2, 14, "FOOD")]}),
    ("a handful of nuts", {"entities": [(0, 1, "QUANTITY"), (2, 9, "UNIT"), (13, 17, "FOOD")]}),
    
    # More diverse patterns
    ("2 cups of brown rice", {"entities": [(0, 1, "QUANTITY"), (2, 6, "UNIT"), (10, 20, "FOOD")]}),
    ("3 large bananas", {"entities": [(0, 1, "QUANTITY"), (8, 15, "FOOD")]}),
    ("150g grilled chicken", {"entities": [(0, 3, "QUANTITY"), (3, 4, "UNIT"), (13, 20, "FOOD")]}),
    ("a small salad", {"entities": [(0, 1, "QUANTITY"), (8, 13, "FOOD")]}),
    ("4 pieces of sushi", {"entities": [(0, 1, "QUANTITY"), (2, 8, "UNIT"), (12, 17, "FOOD")]}),
    ("1.5 cups of oatmeal", {"entities": [(0, 3, "QUANTITY"), (4, 8, "UNIT"), (12, 19, "FOOD")]}),
    ("250ml orange juice", {"entities": [(0, 3, "QUANTITY"), (3, 5, "UNIT"), (6, 18, "FOOD")]}),
    ("two medium potatoes", {"entities": [(0, 3, "QUANTITY"), (11, 19, "FOOD")]}),
    ("6 oz salmon", {"entities": [(0, 1, "QUANTITY"), (2, 4, "UNIT"), (5, 11, "FOOD")]}),
    ("a plate of pasta", {"entities": [(0, 1, "QUANTITY"), (2, 7, "UNIT"), (11, 16, "FOOD")]}),
    
    # Complex patterns
    ("I had 2 cups of rice and a banana", {"entities": [(6, 7, "QUANTITY"), (8, 12, "UNIT"), (16, 20, "FOOD"), (27, 33, "FOOD")]}),
    ("ate 3 eggs and 2 slices of toast", {"entities": [(4, 5, "QUANTITY"), (6, 10, "FOOD"), (15, 16, "QUANTITY"), (17, 23, "UNIT"), (27, 32, "FOOD")]}),
    ("200g chicken and 100g broccoli", {"entities": [(0, 3, "QUANTITY"), (3, 4, "UNIT"), (5, 12, "FOOD"), (17, 20, "QUANTITY"), (20, 21, "UNIT"), (22, 30, "FOOD")]}),
    
    # Variations with different food items
    ("1 cup of quinoa", {"entities": [(0, 1, "QUANTITY"), (2, 5, "UNIT"), (9, 15, "FOOD")]}),
    ("2 tablespoons of peanut butter", {"entities": [(0, 1, "QUANTITY"), (2, 13, "UNIT"), (17, 30, "FOOD")]}),
    ("a can of tuna", {"entities": [(0, 1, "QUANTITY"), (2, 5, "UNIT"), (9, 13, "FOOD")]}),
    ("3 slices of pizza", {"entities": [(0, 1, "QUANTITY"), (2, 8, "UNIT"), (12, 17, "FOOD")]}),
    ("half cup of almonds", {"entities": [(0, 4, "QUANTITY"), (5, 8, "UNIT"), (12, 19, "FOOD")]}),
    ("one avocado", {"entities": [(0, 3, "QUANTITY"), (4, 11, "FOOD")]}),
    ("2 oz cheese", {"entities": [(0, 1, "QUANTITY"), (2, 4, "UNIT"), (5, 11, "FOOD")]}),
    
    # More vegetables and fruits
    ("1 cup of carrots", {"entities": [(0, 1, "QUANTITY"), (2, 5, "UNIT"), (9, 16, "FOOD")]}),
    ("2 tomatoes", {"entities": [(0, 1, "QUANTITY"), (2, 10, "FOOD")]}),
    ("a bunch of grapes", {"entities": [(0, 1, "QUANTITY"), (2, 7, "UNIT"), (11, 17, "FOOD")]}),
    ("150g blueberries", {"entities": [(0, 3, "QUANTITY"), (3, 4, "UNIT"), (5, 16, "FOOD")]}),
    ("one mango", {"entities": [(0, 3, "QUANTITY"), (4, 9, "FOOD")]}),
    
    # Protein sources
    ("8 oz beef", {"entities": [(0, 1, "QUANTITY"), (2, 4, "UNIT"), (5, 9, "FOOD")]}),
    ("2 servings of tofu", {"entities": [(0, 1, "QUANTITY"), (2, 10, "UNIT"), (14, 18, "FOOD")]}),
    ("a piece of fish", {"entities": [(0, 1, "QUANTITY"), (2, 7, "UNIT"), (11, 15, "FOOD")]}),
    ("4 oz turkey", {"entities": [(0, 1, "QUANTITY"), (2, 4, "UNIT"), (5, 11, "FOOD")]}),
    
    # Dairy and beverages
    ("1 cup of yogurt", {"entities": [(0, 1, "QUANTITY"), (2, 5, "UNIT"), (9, 15, "FOOD")]}),
    ("2 glasses of milk", {"entities": [(0, 1, "QUANTITY"), (2, 9, "UNIT"), (13, 17, "FOOD")]}),
    ("a bottle of water", {"entities": [(0, 1, "QUANTITY"), (2, 8, "UNIT"), (12, 17, "FOOD")]}),
    
    # Snacks and desserts
    ("a handful of chips", {"entities": [(0, 1, "QUANTITY"), (2, 9, "UNIT"), (13, 18, "FOOD")]}),
    ("2 cookies", {"entities": [(0, 1, "QUANTITY"), (2, 9, "FOOD")]}),
    ("one chocolate bar", {"entities": [(0, 3, "QUANTITY"), (4, 17, "FOOD")]}),
]

def train_model(n_iter=50):
    """Train a spaCy NER model with expanded data"""
    # Create a blank 'en' model
    model = spacy.blank("en")
    
    # Create the NER component and add it to the pipeline
    if "ner" not in model.pipe_names:
        ner = model.add_pipe("ner", last=True)
    else:
        ner = model.get_pipe("ner")
    
    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    # Disable other pipes during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in model.pipe_names if pipe not in pipe_exceptions]
    
    with model.disable_pipes(*other_pipes):
        optimizer = model.begin_training()
        
        print(f"Training model for {n_iter} iterations with {len(TRAIN_DATA)} examples...")
        
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            
            # Batch up the examples using spaCy's minibatch
            for text, annotations in TRAIN_DATA:
                doc = model.make_doc(text)
                example = Example.from_dict(doc, annotations)
                model.update([example], drop=0.5, losses=losses)
            
            if (itn + 1) % 10 == 0:
                print(f"Iteration {itn + 1}, Losses: {losses}")

    # Save the model
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    model.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model()
