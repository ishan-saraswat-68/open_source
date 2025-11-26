# Nutri-Vision Text Analysis API

A powerful FastAPI-based service that uses Machine Learning (ML) to extract food items and nutritional information from natural language text. It integrates with the USDA FoodData Central API for accurate nutrition data.

## Features

- **ML-Powered Extraction**: Uses a custom spaCy NER model to identify food items, quantities, and units from natural language text
- **Intelligent Entity Grouping**: Automatically groups related entities (quantity + unit + food) for accurate extraction
- **Dynamic Confidence Scoring**: Provides real confidence scores based on entity detection quality and completeness
- **USDA Integration**: Fetches accurate nutritional data (calories, macros) from the USDA FoodData Central database
- **Smart Fallbacks**: Uses a comprehensive mock nutrition database (50+ foods) when USDA data is unavailable
- **Detailed Analysis**: Returns complete macronutrient breakdowns (protein, carbs, fats, fiber, sugar)
- **Multi-Item Support**: Handles complex meal descriptions with multiple food items in a single request
- **Interactive Web UI**: User-friendly HTML interface for easy testing and usage

## Project Structure

```
.
├── main.py              # FastAPI application entry point
├── app.html             # Frontend web interface
├── ml/                  # Machine Learning components
│   ├── model/           # Trained spaCy NER model
│   ├── spacy_extractor.py # ML extraction logic with confidence scoring
│   └── train_model.py   # Model training script (80+ training examples)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure USDA API (Optional)
Get a free API key from [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html) and set it in your environment:
```bash
export USDA_API_KEY=your_api_key_here
```

Alternatively, create a `.env` file:
```
USDA_API_KEY=your_api_key_here
```

### 3. Train the Model (Optional)
The repository includes a pre-trained model, but you can retrain it for improved accuracy:
```bash
python ml/train_model.py
```

This trains the model with 80+ diverse examples covering various food patterns, quantities, and units.

### 4. Run the Server
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Usage

### Web Interface
Open `app.html` in your browser to use the interactive UI with real-time nutrition analysis.

### API Endpoints

#### `POST /analyze/text`
Analyze a text description of a meal.

**Request:**
```json
{
  "text": "I had 2 eggs and a slice of toast",
  "include_usda": true
}
```

**Response:**
```json
{
  "success": true,
  "input_type": "text",
  "raw_input": "I had 2 eggs and a slice of toast",
  "items": [
    {
      "name": "eggs",
      "quantity": 2.0,
      "unit": "serving",
      "macros": {
        "calories": 310.0,
        "protein": 26.0,
        "carbs": 2.2,
        "fats": 22.0
      },
      "confidence": 0.95,
      "source": "text_usda"
    }
  ],
  "totals": {
    "calories": 575.0,
    "protein": 35.0,
    "carbs": 51.2,
    "fats": 25.2
  },
  "processing_time": 1.234,
  "warnings": [],
  "metadata": {
    "ml_available": true,
    "usda_configured": true
  }
}
```

#### `GET /health`
Check API health and service status.

#### `GET /config`
Get current configuration and service availability.

#### `GET /`
API information and available endpoints.

## ML Model Details

### Named Entity Recognition (NER)
The project uses a custom spaCy NER model trained to identify three entity types:

- **FOOD**: Food items (e.g., "apple", "chicken breast", "brown rice")
- **QUANTITY**: Numerical quantities including fractions (e.g., "2", "1.5", "half", "quarter")
- **UNIT**: Units of measurement (e.g., "cup", "grams", "slice", "oz", "tablespoons")

### Confidence Scoring
The model provides dynamic confidence scores based on:
- **0.95**: All three components detected (quantity + unit + food)
- **0.80**: Food + one other component (quantity or unit)
- **0.65**: Only food detected
- **0.50**: Low confidence fallback

Confidence is further boosted (+0.05) when USDA verification is successful.

### Entity Grouping
The extractor intelligently groups related entities:
- Proximity-based grouping
- Automatic separation of multiple food items
- Smart matching of quantities and units to their corresponding foods

### Training Data
The model is trained on 80+ diverse examples covering:
- Basic patterns (single food items)
- Complex patterns (multiple items in one sentence)
- Various food categories (fruits, vegetables, proteins, carbs, dairy, snacks)
- Different quantity formats (numbers, fractions, words)
- Multiple unit types (metric, imperial, serving sizes)

## Mock Nutrition Database

When USDA data is unavailable, the system uses a comprehensive mock database with 50+ foods including:
- **Fruits**: apple, banana, orange, strawberry, mango, etc.
- **Vegetables**: broccoli, carrot, tomato, spinach, bell pepper, etc.
- **Proteins**: chicken, beef, fish, salmon, eggs, tofu, etc.
- **Carbohydrates**: rice, bread, pasta, potato, oats, quinoa, etc.
- **Dairy & Snacks**: cheese, yogurt, milk, nuts, almonds, avocado, etc.
- **Popular Dishes**: pizza, burger, salad, sandwich, soup, wrap, etc.

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **spaCy**: Industrial-strength NLP library for entity recognition
- **USDA FoodData Central API**: Official USDA nutrition database
- **Pydantic**: Data validation and settings management
- **httpx**: Async HTTP client for USDA API calls

## Development

### Project Version
Current version: **3.0.0**

### Requirements
- Python 3.7+
- spaCy 3.x
- FastAPI
- See `requirements.txt` for complete list

## License

MIT
