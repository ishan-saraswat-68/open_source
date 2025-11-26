# Nutri-Vision Text Analysis API

A powerful FastAPI-based service that uses Machine Learning (ML) to extract food items and nutritional information from natural language text. It integrates with the USDA FoodData Central API for accurate nutrition data.

## Features

- **ML-Powered Extraction**: Uses a custom spaCy model to identify food items, quantities, and units from text.
- **USDA Integration**: Fetches accurate nutritional data (calories, macros) from the USDA database.
- **Smart Fallbacks**: Uses a comprehensive mock database if USDA data is unavailable.
- **Detailed Analysis**: Returns macronutrient breakdowns (protein, carbs, fats, fiber, sugar).
- **Confidence Scores**: Provides confidence levels for extracted items.

## Project Structure

```
.
├── main.py              # FastAPI application entry point
├── app.html             # Frontend interface
├── ml/                  # Machine Learning components
│   ├── model/           # Trained spaCy model
│   ├── spacy_extractor.py # ML extraction logic
│   └── train_model.py   # Script to train the model
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Configure USDA API (Optional)**
   Get a free API key from [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html) and set it in your environment:
   ```bash
   export USDA_API_KEY=your_api_key_here
   ```

3. **Train the Model (Optional)**
   If you want to improve the extraction accuracy, you can retrain the model:
   ```bash
   python ml/train_model.py
   ```

4. **Run the Server**
   ```bash
   uvicorn main:app --reload
   ```

## Usage

### Web Interface
Open `app.html` in your browser to use the interactive UI.

### API Endpoints

- `POST /analyze/text`: Analyze a text description of a meal.
  ```json
  {
    "text": "I had 2 eggs and a slice of toast",
    "include_usda": true
  }
  ```

- `GET /health`: Check API health and service status.
- `GET /config`: Get current configuration.

## ML Model

The project uses a custom Named Entity Recognition (NER) model trained with spaCy to identify:
- **FOOD**: Food items (e.g., "apple", "chicken breast")
- **QUANTITY**: Numerical quantities (e.g., "2", "1.5", "half")
- **UNIT**: Units of measurement (e.g., "cup", "grams", "slice")

The model is located in `ml/model`.

## License

MIT
