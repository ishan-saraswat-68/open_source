import spacy
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from word2number import w2n

# Path to the trained model
MODEL_DIR = Path(__file__).parent / "model"

# Load model if it exists, otherwise return None
try:
    nlp = spacy.load(MODEL_DIR)
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load spaCy model from {MODEL_DIR}: {e}")
    MODEL_LOADED = False
    nlp = None

# Unit normalization mapping
UNIT_ALIASES = {
    "g": "grams", "gram": "grams", "grams": "grams",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml",
    "l": "liters", "liter": "liters", "litre": "liters", "litres": "liters",
    "cup": "cups", "cups": "cups",
    "slice": "slices", "slices": "slices",
    "piece": "pieces", "pieces": "pieces", "pc": "pieces",
    "tbsp": "tablespoons", "tablespoon": "tablespoons", "tablespoons": "tablespoons",
    "tsp": "teaspoons", "teaspoon": "teaspoons", "teaspoons": "teaspoons",
    "glass": "glasses", "glasses": "glasses",
    "serving": "servings", "servings": "servings", "serve": "servings",
    "oz": "ounces", "ounce": "ounces", "ounces": "ounces",
    "lb": "pounds", "pound": "pounds", "pounds": "pounds",
    "bowl": "bowls", "bowls": "bowls",
    "plate": "plates", "plates": "plates",
    "can": "cans", "cans": "cans",
    "bottle": "bottles", "bottles": "bottles",
    "handful": "handful", "bunch": "bunch",
}

def parse_number(qty_str: str) -> float:
    """Convert quantity string to float"""
    if not qty_str:
        return 1.0
    
    qty_str = qty_str.strip().lower()
    
    try:
        return float(qty_str)
    except ValueError:
        pass
        
    if "/" in qty_str:
        try:
            parts = qty_str.split("/")
            return float(parts[0]) / float(parts[1])
        except:
            pass
    
    # Handle mixed fractions like "1 1/2"
    mixed_match = re.match(r'(\d+)\s+(\d+)/(\d+)', qty_str)
    if mixed_match:
        whole, num, denom = mixed_match.groups()
        try:
            return float(whole) + (float(num) / float(denom))
        except ZeroDivisionError:
            pass
            
    # Word to number
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "half": 0.5, "quarter": 0.25, "third": 0.33,
        "a": 1, "an": 1
    }
    
    if qty_str in word_numbers:
        return float(word_numbers[qty_str])
    
    try:
        return float(w2n.word_to_num(qty_str))
    except:
        pass
        
    return 1.0

def normalize_unit(unit: str) -> str:
    """Normalize unit to standard form"""
    if not unit:
        return "servings"
    
    unit_lower = unit.lower().strip()
    return UNIT_ALIASES.get(unit_lower, unit_lower)

def calculate_confidence(entities: List[Tuple]) -> float:
    """
    Calculate confidence based on entity scores.
    Returns average confidence of all entities in the group.
    """
    if not entities:
        return 0.5
    
    # Get average of entity scores (if available)
    # spaCy entities have a score attribute in some models
    scores = []
    for ent, score in entities:
        if score is not None:
            scores.append(score)
    
    if scores:
        return sum(scores) / len(scores)
    
    # Fallback: base confidence on entity types present
    entity_types = set(label for _, _, label in [(e.text, e.start, e.label_) for e, _ in entities])
    
    if "FOOD" in entity_types:
        if "QUANTITY" in entity_types and "UNIT" in entity_types:
            return 0.95  # All three components present
        elif "QUANTITY" in entity_types or "UNIT" in entity_types:
            return 0.80  # Food + one other component
        else:
            return 0.65  # Only food detected
    
    return 0.50  # Low confidence if no food detected

def group_entities_by_proximity(entities: List) -> List[List]:
    """
    Group entities that are close to each other.
    Creates a new group when encountering a FOOD entity after already having one.
    """
    if not entities:
        return []
    
    groups = []
    current_group = []
    has_food_in_group = False
    
    for i, ent in enumerate(entities):
        # If we encounter a FOOD entity and already have one, start new group
        if ent.label_ == "FOOD" and has_food_in_group:
            # Save current group
            if current_group:
                groups.append(current_group)
            # Start new group with this food
            current_group = [ent]
            has_food_in_group = True
        else:
            # Add to current group
            current_group.append(ent)
            if ent.label_ == "FOOD":
                has_food_in_group = True
    
    # Add the last group
    if current_group:
        groups.append(current_group)
    
    return groups

def extract_item_from_group(group: List) -> Dict[str, Any]:
    """
    Extract a food item from a group of entities.
    Returns a dict with ingredient, quantity, unit, and confidence.
    """
    quantity = 1.0
    unit = "serving"
    ingredient = ""
    entity_info = []
    
    # Separate entities by type
    quantities = []
    units = []
    foods = []
    
    for ent in group:
        score = getattr(ent, 'score', None) or getattr(ent, 'confidence', None)
        entity_info.append((ent, score))
        
        if ent.label_ == "QUANTITY":
            quantities.append(ent)
        elif ent.label_ == "UNIT":
            units.append(ent)
        elif ent.label_ == "FOOD":
            foods.append(ent)
    
    # Build ingredient name from all FOOD entities
    if foods:
        ingredient = " ".join(f.text for f in foods)
    
    # Match quantity and unit to food
    # Strategy: Use the closest QUANTITY and UNIT that appear BEFORE the first FOOD
    if foods:
        first_food_start = foods[0].start
        
        # Find quantity before food
        valid_quantities = [q for q in quantities if q.end <= first_food_start]
        if valid_quantities:
            # Use the closest one
            closest_qty = max(valid_quantities, key=lambda q: q.end)
            quantity = parse_number(closest_qty.text)
        
        # Find unit before food
        valid_units = [u for u in units if u.end <= first_food_start]
        if valid_units:
            # Use the closest one
            closest_unit = max(valid_units, key=lambda u: u.end)
            unit = normalize_unit(closest_unit.text)
    
    # Calculate confidence for this item
    confidence = calculate_confidence(entity_info)
    
    return {
        "ingredient": ingredient.strip(),
        "quantity": quantity,
        "unit": unit,
        "confidence": round(confidence, 2)
    }

def spacy_extract(text: str) -> List[Dict[str, Any]]:
    """
    Extract food items using the trained spaCy NER model.
    Returns list of items with real confidence scores.
    """
    if not MODEL_LOADED or not nlp:
        print("SpaCy model not loaded, returning empty list.")
        return []

    doc = nlp(text)
    
    # Get all entities
    entities = list(doc.ents)
    
    if not entities:
        return []
    
    # Group entities by proximity
    entity_groups = group_entities_by_proximity(entities)
    
    # Extract items from each group
    items = []
    for group in entity_groups:
        # Check if group has at least a FOOD entity
        has_food = any(ent.label_ == "FOOD" for ent in group)
        
        if has_food:
            item = extract_item_from_group(group)
            if item["ingredient"]:  # Only add if we have an ingredient
                items.append(item)
    
    return items
