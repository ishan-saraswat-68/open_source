#!/bin/bash

# Complete setup script for Nutri-Vision Text Analysis API
# Run this script to create all folders and files

# Create project root directory
mkdir -p nutri-vision-text-api
cd nutri-vision-text-api

# Create directory structure
mkdir -p nlp

# Create __init__.py files for Python packages
touch nlp/__init__.py

# Create main application files
cat > main.py << 'EOF'
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import os
import time

# Import NLP components
try:
    from nlp.enhanced_extractor import enhanced_extract
    NLP_AVAILABLE = True
except ImportError as e:
    print(f"NLP import failed: {e}")
    NLP_AVAILABLE = False
    from nlp.simple_extractor import simple_extract as enhanced_extract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nutri-Vision Text Analysis API",
    description="Enhanced text-based nutrition analysis with USDA integration",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
USDA_API_KEY = os.getenv('USDA_API_KEY', 'ecXV1I6dbQEUodkjrsfklpCMVLRHdT4E5f7wvELk')
USDA_BASE_URL = 'https://api.nal.usda.gov/fdc/v1'

# Models
class MacroInfo(BaseModel):
    calories: float = Field(default=0.0, ge=0)
    protein: float = Field(default=0.0, ge=0)
    carbs: float = Field(default=0.0, ge=0)
    fats: float = Field(default=0.0, ge=0)
    fiber: Optional[float] = Field(default=None, ge=0)
    sugar: Optional[float] = Field(default=None, ge=0)

class FoodItem(BaseModel):
    name: str
    quantity: float = Field(default=1.0, gt=0)
    unit: str = Field(default="serving")
    macros: MacroInfo
    confidence: Optional[float] = Field(None, ge=0, le=1)
    source: str = Field(default="api")
    notes: Optional[str] = None
    usda_food_id: Optional[str] = None

class NutritionAnalysis(BaseModel):
    success: bool
    input_type: str
    raw_input: str
    items: List[FoodItem] = []
    totals: MacroInfo
    processing_time: Optional[float] = None
    warnings: List[str] = []
    metadata: dict = {}

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_usda: bool = Field(True)

class VoiceAnalysisRequest(BaseModel):
    transcribed_text: str = Field(..., min_length=1, max_length=5000)
    include_usda: bool = Field(True)

# USDA API functions
async def search_usda_food(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    try:
        if not USDA_API_KEY or USDA_API_KEY == 'your_usda_api_key_here':
            logger.warning("USDA API key not configured")
            return []
            
        params = {
            'api_key': USDA_API_KEY,
            'query': query,
            'pageSize': limit,
            'dataType': ['Foundation', 'SR Legacy', 'Survey (FNDDS)']
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{USDA_BASE_URL}/foods/search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('foods', [])
            else:
                logger.error(f"USDA search failed: {response.status_code}")
                return []
                
    except Exception as e:
        logger.error(f"USDA search error: {str(e)}")
        return []

async def get_usda_nutrition(food_id: str) -> Optional[Dict[str, Any]]:
    try:
        if not USDA_API_KEY or USDA_API_KEY == 'your_usda_api_key_here':
            return None
            
        params = {'api_key': USDA_API_KEY}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{USDA_BASE_URL}/food/{food_id}", params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"USDA nutrition lookup failed: {response.status_code}")
                return None
                
    except Exception as e:
        logger.error(f"USDA nutrition error: {str(e)}")
        return None

def extract_usda_macros(usda_food: Dict[str, Any]) -> MacroInfo:
    nutrients = {}
    
    nutrient_map = {
        'Energy': 'calories',
        'Protein': 'protein', 
        'Carbohydrate, by difference': 'carbs',
        'Total lipid (fat)': 'fats',
        'Fiber, total dietary': 'fiber',
        'Sugars, total including NLEA': 'sugar'
    }
    
    for nutrient in usda_food.get('foodNutrients', []):
        nutrient_name = nutrient.get('nutrient', {}).get('name', '')
        
        if nutrient_name in nutrient_map:
            value = nutrient.get('amount', 0)
            if nutrient_name == 'Energy':
                unit = nutrient.get('nutrient', {}).get('unitName', '').upper()
                if unit == 'KCAL':
                    nutrients[nutrient_map[nutrient_name]] = float(value)
                elif unit == 'KJ':
                    nutrients[nutrient_map[nutrient_name]] = float(value) / 4.184
            else:
                nutrients[nutrient_map[nutrient_name]] = float(value)
    
    return MacroInfo(
        calories=nutrients.get('calories', 0.0),
        protein=nutrients.get('protein', 0.0),
        carbs=nutrients.get('carbs', 0.0),
        fats=nutrients.get('fats', 0.0),
        fiber=nutrients.get('fiber'),
        sugar=nutrients.get('sugar')
    )

def get_mock_nutrition_by_food_name(food_name: str) -> dict:
    food_name_lower = food_name.lower()
    
    nutrition_db = {
        "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fats": 0.3, "fiber": 4.0, "sugar": 19},
        "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fats": 0.4, "fiber": 3.1, "sugar": 14},
        "orange": {"calories": 65, "protein": 1.3, "carbs": 16, "fats": 0.2, "fiber": 3.4, "sugar": 13},
        "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6, "fiber": 0, "sugar": 0},
        "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6, "fiber": 0, "sugar": 0},
        "beef": {"calories": 250, "protein": 26, "carbs": 0, "fats": 15, "fiber": 0, "sugar": 0},
        "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fats": 0.3, "fiber": 0.4, "sugar": 0.1},
        "bread": {"calories": 265, "protein": 9, "carbs": 49, "fats": 3.2, "fiber": 2.7, "sugar": 5.0},
        "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fats": 11, "fiber": 0, "sugar": 1.1},
        "milk": {"calories": 61, "protein": 3.2, "carbs": 4.8, "fats": 3.3, "fiber": 0, "sugar": 5.1},
        "cheese": {"calories": 113, "protein": 7, "carbs": 1, "fats": 9, "fiber": 0, "sugar": 0.5},
        "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fats": 0.4, "fiber": 0, "sugar": 3.6},
        "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fats": 1.1, "fiber": 1.8, "sugar": 0.6},
        "potato": {"calories": 77, "protein": 2.0, "carbs": 17, "fats": 0.1, "fiber": 2.1, "sugar": 0.8},
        "broccoli": {"calories": 55, "protein": 4.6, "carbs": 11, "fats": 0.6, "fiber": 5.1, "sugar": 2.6},
        "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fats": 0.2, "fiber": 2.8, "sugar": 4.7},
        "tomato": {"calories": 22, "protein": 1.1, "carbs": 4.8, "fats": 0.2, "fiber": 1.4, "sugar": 3.2},
        "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fats": 13, "fiber": 0, "sugar": 0},
        "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fats": 10, "fiber": 2.3, "sugar": 3.8},
        "burger": {"calories": 295, "protein": 17, "carbs": 23, "fats": 14, "fiber": 2.0, "sugar": 4.0},
        "salad": {"calories": 65, "protein": 5, "carbs": 7, "fats": 4, "fiber": 3.0, "sugar": 4.0},
    }
    
    if food_name_lower in nutrition_db:
        return nutrition_db[food_name_lower]
    
    for key, nutrition in nutrition_db.items():
        if key in food_name_lower or food_name_lower in key:
            return nutrition
    
    return {"calories": 150, "protein": 8, "carbs": 20, "fats": 5, "fiber": 2.0, "sugar": 5.0}

async def process_text_analysis(text: str, include_usda: bool = True) -> List[FoodItem]:
    try:
        logger.info(f"Processing text: {text[:100]}...")
        extracted_items = enhanced_extract(text)
        
    except Exception as e:
        logger.error(f"NLP extraction failed: {e}")
        return []
    
    if not extracted_items:
        return []
    
    processed_items = []
    
    for item in extracted_items:
        try:
            item_name = item.get("ingredient", "unknown")
            item_quantity = float(item.get("quantity", 1.0))
            item_unit = item.get("unit", "serving")
            
            macros = None
            notes = []
            usda_food_id = None
            
            if include_usda and USDA_API_KEY and USDA_API_KEY != 'your_usda_api_key_here':
                try:
                    usda_results = await search_usda_food(item_name, limit=1)
                    if usda_results:
                        usda_food = usda_results[0]
                        usda_food_id = str(usda_food.get('fdcId', ''))
                        
                        usda_detail = await get_usda_nutrition(usda_food_id)
                        if usda_detail:
                            macros = extract_usda_macros(usda_detail)
                            macros.calories *= item_quantity
                            macros.protein *= item_quantity
                            macros.carbs *= item_quantity
                            macros.fats *= item_quantity
                            if macros.fiber: macros.fiber *= item_quantity
                            if macros.sugar: macros.sugar *= item_quantity
                        else:
                            notes.append("USDA nutrition lookup failed")
                    else:
                        notes.append("No USDA match found")
                        
                except Exception as e:
                    logger.error(f"USDA processing error: {e}")
                    notes.append(f"USDA error: {str(e)}")
            
            if not macros:
                mock_nutrition = get_mock_nutrition_by_food_name(item_name)
                scaled_nutrition = {k: v * item_quantity for k, v in mock_nutrition.items()}
                macros = MacroInfo(**scaled_nutrition)
                notes.append("Using estimated nutrition values")
            
            food_item = FoodItem(
                name=item_name,
                quantity=item_quantity,
                unit=item_unit,
                macros=macros,
                confidence=0.85 if usda_food_id else 0.6,
                source="text_usda" if usda_food_id else "text_estimated",
                notes="; ".join(notes) if notes else None,
                usda_food_id=usda_food_id
            )
            
            processed_items.append(food_item)
            
        except Exception as e:
            logger.error(f"Error processing item {item}: {str(e)}")
            continue
    
    return processed_items

def calculate_totals(items: List[FoodItem]) -> MacroInfo:
    totals = MacroInfo(
        calories=sum(item.macros.calories for item in items),
        protein=sum(item.macros.protein for item in items),
        carbs=sum(item.macros.carbs for item in items),
        fats=sum(item.macros.fats for item in items),
        fiber=sum((item.macros.fiber or 0) for item in items),
        sugar=sum((item.macros.sugar or 0) for item in items)
    )
    
    totals.calories = round(totals.calories, 1)
    totals.protein = round(totals.protein, 1) 
    totals.carbs = round(totals.carbs, 1)
    totals.fats = round(totals.fats, 1)
    if totals.fiber: totals.fiber = round(totals.fiber, 1)
    if totals.sugar: totals.sugar = round(totals.sugar, 1)
    
    return totals

# API Endpoints
@app.get("/")
async def root():
    return {
        "name": "Nutri-Vision Text Analysis API",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "text": "/analyze/text",
            "voice": "/analyze/voice",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "services": {
            "api": "active",
            "nlp": "available" if NLP_AVAILABLE else "basic",
            "usda": "configured" if USDA_API_KEY != 'your_usda_api_key_here' else "needs_token"
        }
    }

@app.post("/analyze/text", response_model=NutritionAnalysis)
async def analyze_text(request: TextAnalysisRequest):
    start_time = time.time()
    
    try:
        items = await process_text_analysis(request.text, request.include_usda)
        totals = calculate_totals(items)
        
        warnings = []
        if not items:
            warnings.append("No food items could be identified")
        
        return NutritionAnalysis(
            success=True,
            input_type="text",
            raw_input=request.text,
            items=items,
            totals=totals,
            processing_time=round(time.time() - start_time, 3),
            warnings=warnings,
            metadata={
                "nlp_available": NLP_AVAILABLE,
                "items_with_usda": sum(1 for item in items if item.usda_food_id)
            }
        )
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        return NutritionAnalysis(
            success=False,
            input_type="text",
            raw_input=request.text,
            items=[],
            totals=MacroInfo(),
            processing_time=round(time.time() - start_time, 3),
            warnings=[f"Analysis failed: {str(e)}"]
        )

@app.post("/analyze/voice", response_model=NutritionAnalysis)
async def analyze_voice(request: VoiceAnalysisRequest):
    start_time = time.time()
    
    try:
        items = await process_text_analysis(request.transcribed_text, request.include_usda)
        totals = calculate_totals(items)
        
        warnings = ["Voice transcription processed as text"]
        if not items:
            warnings.append("No food items could be identified")
        
        return NutritionAnalysis(
            success=True,
            input_type="voice",
            raw_input=request.transcribed_text,
            items=items,
            totals=totals,
            processing_time=round(time.time() - start_time, 3),
            warnings=warnings
        )
        
    except Exception as e:
        logger.error(f"Voice analysis failed: {str(e)}")
        return NutritionAnalysis(
            success=False,
            input_type="voice",
            raw_input=request.transcribed_text,
            items=[],
            totals=MacroInfo(),
            processing_time=round(time.time() - start_time, 3),
            warnings=[f"Analysis failed: {str(e)}"]
        )

@app.get("/config")
async def get_configuration():
    return {
        "version": "3.0.0",
        "services": {
            "nlp": {"available": NLP_AVAILABLE},
            "usda": {"configured": USDA_API_KEY != 'your_usda_api_key_here'}
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting Nutri-Vision Text Analysis API v3.0.0")
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF

# Create enhanced NLP extractor
cat > nlp/enhanced_extractor.py << 'EOF'
import re
from typing import List, Dict
try:
    from word2number import w2n
except ImportError:
    w2n = None

UNIT_ALIASES = {
    "g": "grams", "gram": "grams", "grams": "grams",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "l": "liters", "cup": "cups", "cups": "cups",
    "slice": "slices", "slices": "slices",
    "piece": "pieces", "pieces": "pieces",
    "tbsp": "tablespoons", "tsp": "teaspoons",
    "glass": "glasses", "serving": "servings",
    "oz": "ounces", "lb": "pounds", "bowl": "bowls"
}

FOOD_DATABASE = {
    "chicken": ["chicken", "chicken breast", "grilled chicken"],
    "beef": ["beef", "steak"],
    "fish": ["fish", "salmon", "tuna"],
    "egg": ["egg", "eggs"],
    "rice": ["rice", "brown rice"],
    "bread": ["bread", "toast"],
    "pasta": ["pasta", "noodles"],
    "apple": ["apple", "apples"],
    "banana": ["banana", "bananas"],
    "orange": ["orange", "oranges"],
    "milk": ["milk"],
    "cheese": ["cheese"],
    "yogurt": ["yogurt"],
    "potato": ["potato", "potatoes"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "tomato": ["tomato", "tomatoes"],
    "salad": ["salad"],
    "pizza": ["pizza"],
    "burger": ["burger"]
}

FOOD_LOOKUP = {}
for category, variations in FOOD_DATABASE.items():
    for variation in variations:
        FOOD_LOOKUP[variation.lower()] = category

def parse_number(qty_str: str) -> float:
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
    
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "half": 0.5, "quarter": 0.25, "a": 1, "an": 1
    }
    
    if qty_str in word_numbers:
        return float(word_numbers[qty_str])
    
    if w2n:
        try:
            return float(w2n.word_to_num(qty_str))
        except:
            pass
    
    return 1.0

def normalize_unit(unit: str) -> str:
    if not unit:
        return "servings"
    return UNIT_ALIASES.get(unit.lower().strip(), unit.lower())

def identify_food(text: str) -> str:
    text_lower = text.lower().strip()
    
    if text_lower in FOOD_LOOKUP:
        return FOOD_LOOKUP[text_lower]
    
    for food_variant, food_name in FOOD_LOOKUP.items():
        if food_variant in text_lower:
            return food_name
    
    return text_lower

def extract_with_patterns(text: str) -> List[Dict]:
    results = []
    
    pattern1 = re.compile(
        r'(?P<qty>[\d.]+|one|two|three|four|five|half|a|an)\s+'
        r'(?P<unit>g|kg|cup|cups|slice|slices|piece|pieces|glass|serving)s?\s+'
        r'(?:of\s+)?'
        r'(?P<ingredient>[a-zA-Z\s]+?)(?=\s*(?:and|with|,|$))',
        re.I
    )
    
    pattern2 = re.compile(
        r'(?P<qty>[\d.]+|one|two|three|four|five|half|a|an)\s+'
        r'(?P<ingredient>[a-zA-Z\s]+?)\s*'
        r'(?P<unit>g|kg|cup|cups|slice|slices)?'
        r'(?=\s*(?:and|with|,|$))',
        re.I
    )
    
    for pattern in [pattern1, pattern2]:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            ingredient = groups.get("ingredient", "").strip()
            
            if not ingredient or len(ingredient) < 2:
                continue
            
            quantity = parse_number(groups.get("qty", "1"))
            unit = normalize_unit(groups.get("unit", "servings"))
            food_name = identify_food(ingredient)
            
            results.append({
                "ingredient": food_name,
                "quantity": quantity,
                "unit": unit
            })
    
    return results

def consolidate_items(items: List[Dict]) -> List[Dict]:
    if not items:
        return []
    
    consolidated = {}
    
    for item in items:
        ingredient = item["ingredient"].lower()
        unit = item["unit"]
        key = f"{ingredient}_{unit}"
        
        if key in consolidated:
            consolidated[key]["quantity"] += item["quantity"]
        else:
            consolidated[key] = {
                "ingredient": ingredient,
                "quantity": item["quantity"],
                "unit": unit
            }
    
    for item in consolidated.values():
        item["quantity"] = round(item["quantity"], 2)
    
    return list(consolidated.values())

def enhanced_extract(text: str) -> List[Dict]:
    if not text or not text.strip():
        return []
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^(i\s+had|i\s+ate|for\s+breakfast|for\s+lunch|for\s+dinner)\s+', '', text, flags=re.I)
    
    extracted = extract_with_patterns(text)
    
    if not extracted:
        parts = re.split(r'[,;]|\s+and\s+', text, flags=re.I)
        for part in parts:
            part = part.strip()
            if len(part) >= 2:
                part_results = extract_with_patterns(part)
                extracted.extend(part_results)
    
    final_items = consolidate_items(extracted)
    return final_items
EOF

# Create simple extractor
cat > nlp/simple_extractor.py << 'EOF'
import re

COMMON_FOODS = {
    "apple", "banana", "orange", "chicken", "beef", "rice", 
    "bread", "egg", "milk", "cheese", "pasta", "pizza"
}

def simple_extract(text: str) -> list:
    if not text:
        return []
    
    text = text.lower()
    results = []
    
    pattern = r'(\d+(?:\.\d+)?)\s+([a-z]+)'
    matches = re.findall(pattern, text)
    
    for qty_str, food in matches:
        if food in COMMON_FOODS:
            try:
                quantity = float(qty_str)
                results.append({
                    "ingredient": food,
                    "quantity": quantity,
                    "unit": "servings"
                })
            except ValueError:
                pass
    
    if not results:
        for food in COMMON_FOODS:
            if food in text:
                results.append({
                    "ingredient": food,
                    "quantity": 1.0,
                    "unit": "servings"
                })
    
    return results if results else [{
        "ingredient": "mixed food",
        "quantity": 1.0,
        "unit": "servings"
    }]
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
httpx==0.25.1
pydantic==2.5.0
word2number==1.1
EOF

# Create .env file
cat > .env << 'EOF'
USDA_API_KEY=ecXV1I6dbQEUodkjrsfklpCMVLRHdT4E5f7wvELk
PORT=8000
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv
.env
*.log
.DS_Store
EOF

# Create README.md
cat > README.md << 'EOF'
# Nutri-Vision Text Analysis API v3.0.0

Text-based nutrition analysis with enhanced NLP and USDA integration.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set USDA API key (optional):
   ```bash
   export USDA_API_KEY="your_key_here"
   ```

3. Run the API:
   ```bash
   python main.py
   ```

4. Open `frontend.html` in your browser

## Features

- Enhanced NLP extraction
- USDA nutrition database
- Voice-to-text support
- Comprehensive food database
- Modern web interface

## Example Usage

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "2 apples and 1 banana", "include_usda": true}'
```

For full documentation, see the complete README in the project files.
EOF

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "📁 Directory structure:"
echo "nutri-vision-text-api/"
echo "├── main.py"
echo "├── nlp/"
echo "│   ├── __init__.py"
echo "│   ├── enhanced_extractor.py"
echo "│   └── simple_extractor.py"
echo "├── requirements.txt"
echo "├── .env"
echo "├── .gitignore"
echo "└── README.md"
echo ""
echo "🚀 Next steps:"
echo "1. cd nutri-vision-text-api"
echo "2. pip install -r requirements.txt"
echo "3. python main.py"
echo ""
echo "Note: You still need to create frontend.html manually from the artifact provided."
EOF

chmod +x setup.sh
echo "✅ Setup script created!"
