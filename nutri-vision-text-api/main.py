import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import os
import time

# Import ML components
try:
    from ml.spacy_extractor import spacy_extract as enhanced_extract
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML import failed: {e}")
    ML_AVAILABLE = False
    
    # Fallback: create a dummy function
    def enhanced_extract(text):
        return []

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

# =============================================================================
# CONFIGURATION
# =============================================================================

USDA_API_KEY = os.getenv('USDA_API_KEY', 'ecXV1I6dbQEUodkjrsfklpCMVLRHdT4E5f7wvELk')
USDA_BASE_URL = 'https://api.nal.usda.gov/fdc/v1'

# =============================================================================
# MODELS
# =============================================================================

class MacroInfo(BaseModel):
    """Nutritional macronutrient information"""
    calories: float = Field(default=0.0, ge=0)
    protein: float = Field(default=0.0, ge=0)
    carbs: float = Field(default=0.0, ge=0)
    fats: float = Field(default=0.0, ge=0)
    fiber: Optional[float] = Field(default=None, ge=0)
    sugar: Optional[float] = Field(default=None, ge=0)

class FoodItem(BaseModel):
    """Individual food item with nutrition"""
    name: str
    quantity: float = Field(default=1.0, gt=0)
    unit: str = Field(default="serving")
    macros: MacroInfo
    confidence: Optional[float] = Field(None, ge=0, le=1)
    source: str = Field(default="api")
    notes: Optional[str] = None
    usda_food_id: Optional[str] = None

class NutritionAnalysis(BaseModel):
    """Complete nutrition analysis response"""
    success: bool
    input_type: str
    raw_input: str
    items: List[FoodItem] = []
    totals: MacroInfo
    processing_time: Optional[float] = None
    warnings: List[str] = []
    metadata: dict = {}

class TextAnalysisRequest(BaseModel):
    """Text analysis request"""
    text: str = Field(..., min_length=1, max_length=5000)
    include_usda: bool = Field(True)



# =============================================================================
# USDA API INTEGRATION
# =============================================================================

async def search_usda_food(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search USDA FoodData Central"""
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
    """Get detailed nutrition from USDA"""
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
    """Extract macronutrients from USDA data"""
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

# =============================================================================
# ENHANCED MOCK NUTRITION DATABASE
# =============================================================================

def get_mock_nutrition_by_food_name(food_name: str) -> dict:
    """Comprehensive mock nutrition database"""
    food_name_lower = food_name.lower()
    
    nutrition_db = {
        # Fruits
        "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fats": 0.3, "fiber": 4.0, "sugar": 19},
        "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fats": 0.4, "fiber": 3.1, "sugar": 14},
        "orange": {"calories": 65, "protein": 1.3, "carbs": 16, "fats": 0.2, "fiber": 3.4, "sugar": 13},
        "strawberry": {"calories": 49, "protein": 1.0, "carbs": 12, "fats": 0.5, "fiber": 3.3, "sugar": 7},
        "grape": {"calories": 69, "protein": 0.7, "carbs": 18, "fats": 0.2, "fiber": 0.9, "sugar": 15},
        "mango": {"calories": 99, "protein": 1.4, "carbs": 25, "fats": 0.6, "fiber": 2.6, "sugar": 23},
        "pineapple": {"calories": 82, "protein": 0.9, "carbs": 22, "fats": 0.2, "fiber": 2.3, "sugar": 16},
        "watermelon": {"calories": 46, "protein": 0.9, "carbs": 12, "fats": 0.2, "fiber": 0.6, "sugar": 9},
        "peach": {"calories": 58, "protein": 1.4, "carbs": 14, "fats": 0.4, "fiber": 2.3, "sugar": 13},
        "pear": {"calories": 101, "protein": 0.6, "carbs": 27, "fats": 0.2, "fiber": 5.5, "sugar": 17},
        
        # Vegetables
        "broccoli": {"calories": 55, "protein": 4.6, "carbs": 11, "fats": 0.6, "fiber": 5.1, "sugar": 2.6},
        "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fats": 0.2, "fiber": 2.8, "sugar": 4.7},
        "tomato": {"calories": 22, "protein": 1.1, "carbs": 4.8, "fats": 0.2, "fiber": 1.4, "sugar": 3.2},
        "lettuce": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fats": 0.2, "fiber": 1.3, "sugar": 0.8},
        "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fats": 0.4, "fiber": 2.2, "sugar": 0.4},
        "cucumber": {"calories": 16, "protein": 0.7, "carbs": 3.6, "fats": 0.1, "fiber": 0.5, "sugar": 1.7},
        "bell pepper": {"calories": 31, "protein": 1.0, "carbs": 6, "fats": 0.3, "fiber": 2.1, "sugar": 4.2},
        "mushroom": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fats": 0.3, "fiber": 1.0, "sugar": 2.0},
        "onion": {"calories": 40, "protein": 1.1, "carbs": 9, "fats": 0.1, "fiber": 1.7, "sugar": 4.2},
        "garlic": {"calories": 149, "protein": 6.4, "carbs": 33, "fats": 0.5, "fiber": 2.1, "sugar": 1.0},
        
        # Proteins
        "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6, "fiber": 0, "sugar": 0},
        "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fats": 3.6, "fiber": 0, "sugar": 0},
        "beef": {"calories": 250, "protein": 26, "carbs": 0, "fats": 15, "fiber": 0, "sugar": 0},
        "pork": {"calories": 242, "protein": 27, "carbs": 0, "fats": 14, "fiber": 0, "sugar": 0},
        "fish": {"calories": 206, "protein": 22, "carbs": 0, "fats": 12, "fiber": 0, "sugar": 0},
        "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fats": 13, "fiber": 0, "sugar": 0},
        "tuna": {"calories": 132, "protein": 28, "carbs": 0, "fats": 1.3, "fiber": 0, "sugar": 0},
        "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fats": 11, "fiber": 0, "sugar": 1.1},
        "tofu": {"calories": 76, "protein": 8, "carbs": 1.9, "fats": 4.8, "fiber": 0.3, "sugar": 0.7},
        
        # Carbohydrates
        "bread": {"calories": 265, "protein": 9, "carbs": 49, "fats": 3.2, "fiber": 2.7, "sugar": 5.0},
        "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fats": 0.3, "fiber": 0.4, "sugar": 0.1},
        "brown rice": {"calories": 112, "protein": 2.3, "carbs": 24, "fats": 0.9, "fiber": 1.8, "sugar": 0.4},
        "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fats": 1.1, "fiber": 1.8, "sugar": 0.6},
        "potato": {"calories": 77, "protein": 2.0, "carbs": 17, "fats": 0.1, "fiber": 2.1, "sugar": 0.8},
        "sweet potato": {"calories": 86, "protein": 1.6, "carbs": 20, "fats": 0.1, "fiber": 3.0, "sugar": 4.2},
        "oats": {"calories": 389, "protein": 16.9, "carbs": 66, "fats": 6.9, "fiber": 10.6, "sugar": 0.99},
        "quinoa": {"calories": 120, "protein": 4.4, "carbs": 21, "fats": 1.9, "fiber": 2.8, "sugar": 0.9},
        
        # Popular dishes
        "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fats": 10, "fiber": 2.3, "sugar": 3.8},
        "burger": {"calories": 295, "protein": 17, "carbs": 23, "fats": 14, "fiber": 2.0, "sugar": 4.0},
        "salad": {"calories": 65, "protein": 5, "carbs": 7, "fats": 4, "fiber": 3.0, "sugar": 4.0},
        "sandwich": {"calories": 230, "protein": 10, "carbs": 30, "fats": 8, "fiber": 3.0, "sugar": 4.0},
        "soup": {"calories": 85, "protein": 4, "carbs": 12, "fats": 2.5, "fiber": 2.0, "sugar": 3.0},
        "wrap": {"calories": 245, "protein": 11, "carbs": 32, "fats": 9, "fiber": 2.5, "sugar": 3.5},
        
        # Dairy & snacks
        "cheese": {"calories": 113, "protein": 7, "carbs": 1, "fats": 9, "fiber": 0, "sugar": 0.5},
        "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fats": 0.4, "fiber": 0, "sugar": 3.6},
        "milk": {"calories": 61, "protein": 3.2, "carbs": 4.8, "fats": 3.3, "fiber": 0, "sugar": 5.1},
        "nuts": {"calories": 607, "protein": 20, "carbs": 16, "fats": 54, "fiber": 8.0, "sugar": 4.0},
        "almonds": {"calories": 579, "protein": 21, "carbs": 22, "fats": 50, "fiber": 12.5, "sugar": 4.4},
        "peanuts": {"calories": 567, "protein": 26, "carbs": 16, "fats": 49, "fiber": 8.5, "sugar": 4.7},
        "avocado": {"calories": 160, "protein": 2, "carbs": 9, "fats": 15, "fiber": 7.0, "sugar": 0.7},
    }
    
    # Exact match
    if food_name_lower in nutrition_db:
        return nutrition_db[food_name_lower]
    
    # Partial match
    for key, nutrition in nutrition_db.items():
        if key in food_name_lower or food_name_lower in key:
            return nutrition
    
    # Default
    return {"calories": 150, "protein": 8, "carbs": 20, "fats": 5, "fiber": 2.0, "sugar": 5.0}

# =============================================================================
# TEXT PROCESSING
# =============================================================================

async def process_text_analysis(text: str, include_usda: bool = True) -> List[FoodItem]:
    """Enhanced text processing with USDA integration"""
    try:
        logger.info(f"Processing text: {text[:100]}...")
        extracted_items = enhanced_extract(text)
        
    except Exception as e:
        logger.error(f"ML extraction failed: {e}")
        return []
    
    if not extracted_items:
        return []
    
    processed_items = []
    
    for item in extracted_items:
        try:
            item_name = item.get("ingredient", "unknown")
            item_quantity = float(item.get("quantity", 1.0))
            item_unit = item.get("unit", "serving")
            item_confidence = item.get("confidence", 0.5)  # Get confidence from extractor
            
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
            
            # Use extractor confidence, boost slightly if USDA verified
            final_confidence = min(item_confidence + 0.05, 1.0) if usda_food_id else item_confidence
            
            food_item = FoodItem(
                name=item_name,
                quantity=item_quantity,
                unit=item_unit,
                macros=macros,
                confidence=final_confidence,
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
    """Calculate total macronutrients"""
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

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Nutri-Vision Text Analysis API",
        "version": "3.0.0",
        "description": "Text-based nutrition analysis with enhanced ML",
        "status": "running",
        "services": {
            "ml": "available" if ML_AVAILABLE else "basic_mode",
            "usda": "configured" if USDA_API_KEY != 'your_usda_api_key_here' else "not_configured"
        },
        "endpoints": {
            "text": "/analyze/text",
            "health": "/health",
            "config": "/config"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "3.0.0",
        "services": {
            "api": "active",
            "ml": "available" if ML_AVAILABLE else "basic",
            "usda": "configured" if USDA_API_KEY != 'your_usda_api_key_here' else "needs_token"
        }
    }

@app.post("/analyze/text", response_model=NutritionAnalysis)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze food from text description"""
    start_time = time.time()
    
    try:
        logger.info(f"Text analysis: '{request.text}'")
        
        items = await process_text_analysis(request.text, request.include_usda)
        totals = calculate_totals(items)
        
        processing_time = round(time.time() - start_time, 3)
        warnings = []
        
        if not items:
            warnings.append("No food items could be identified")
        
        if not ML_AVAILABLE:
            warnings.append("Using basic extraction (enhanced ML unavailable)")
            
        if not USDA_API_KEY or USDA_API_KEY == 'your_usda_api_key_here':
            warnings.append("Using estimated nutrition data (USDA API not configured)")
        
        return NutritionAnalysis(
            success=True,
            input_type="text",
            raw_input=request.text,
            items=items,
            totals=totals,
            processing_time=processing_time,
            warnings=warnings,
            metadata={
                "ml_available": ML_AVAILABLE,
                "usda_configured": USDA_API_KEY != 'your_usda_api_key_here',
                "usda_lookup_enabled": request.include_usda,
                "items_with_usda": sum(1 for item in items if item.usda_food_id),
                "items_estimated": sum(1 for item in items if not item.usda_food_id)
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
            warnings=[f"Analysis failed: {str(e)}"],
            metadata={"error": str(e)}
        )



@app.get("/config")
async def get_configuration():
    """Get API configuration status"""
    return {
        "version": "3.0.0",
        "services": {
            "ml_module": {
                "available": ML_AVAILABLE,
                "status": "enhanced" if ML_AVAILABLE else "basic"
            },
            "usda_api": {
                "configured": USDA_API_KEY != 'your_usda_api_key_here',
                "status": "ready" if USDA_API_KEY != 'your_usda_api_key_here' else "needs_token",
                "base_url": USDA_BASE_URL
            }
        },
        "features": {
            "text_analysis": True,
            "usda_integration": USDA_API_KEY != 'your_usda_api_key_here'
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("="*60)
    logger.info("Starting Nutri-Vision Text Analysis API v3.0.0")
    logger.info("="*60)
    logger.info(f"🚀 Server: http://localhost:{port}")
    logger.info(f"📝 ML: {'✅ Enhanced' if ML_AVAILABLE else '⚠️ Basic mode'}")
    logger.info(f"🥗 USDA: {'✅ Configured' if USDA_API_KEY != 'your_usda_api_key_here' else '❌ Not configured'}")
    logger.info("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)