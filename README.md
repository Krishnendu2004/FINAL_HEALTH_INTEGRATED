# FINAL HEALTH INTEGRATED - Complete Wellness Platform

A comprehensive health and nutrition application that provides personalized diet recommendations, calorie tracking, and sports nutrition guidance for athletes.

## Features

- **Personalized Calorie Calculations**: BMR/TDEE-based calculations with activity level adjustments
- **Position-Specific Diet Plans**: Tailored nutrition for different sports positions (football, basketball, etc.)
- **Macro Nutrient Optimization**: ISSN-based sports nutrition science with position-adjusted macros
- **Food Image Recognition**: CNN-based food identification from uploaded images
- **Meal Planning**: Automated meal generation with balanced nutrient distribution
- **Real-time UI Updates**: Dynamic calculations that respond to user input changes

## Technology Stack

- **Backend**: Flask (Python)
- **ML Models**: TensorFlow/Keras (Calorie Predictor, Diet Recommender, Macro Predictor, Food CNN)
- **Frontend**: HTML/CSS/JavaScript with Bootstrap and Chart.js
- **Data Processing**: Pandas for food database management
- **Image Processing**: OpenCV/PIL for food image analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Krishnendu2004/FINAL_HEALTH_INTEGRATED.git
cd FINAL_HEALTH_INTEGRATED
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open browser to `http://localhost:5000`

## Usage

1. Enter your personal details (age, weight, height, gender, activity level)
2. Select your sport and position
3. Choose dietary preferences and cuisine type
4. Upload food images for recognition or get personalized meal plans
5. View detailed nutrition breakdowns and macro distributions

## Models

- `calorie_predictor.keras`: Predicts daily calorie needs
- `diet_recommender_enhanced.keras`: Recommends diet types
- `macro_predictor_enhanced.keras`: Predicts macro nutrient ratios
- `food_cnn_model.h5`: Classifies food from images (20 classes)

## Data

- `complete_food_dataset.csv`: 408 food items with nutritional data
- `calories_lookup.csv`: Calorie reference data
- `class_names.txt`: Food classification labels

## Professional Features

- **Science-Based Calculations**: Uses ISSN sports nutrition guidelines
- **Position-Specific Nutrition**: Different macro ratios for goalkeepers, midfielders, strikers
- **ML Validation**: Fallback to traditional calculations when ML outputs are unrealistic
- **Real-time Responsiveness**: Auto-calculates on input changes

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── runtime.txt           # Runtime specification
├── templates/
│   └── index.html        # Main UI template
├── static/
│   ├── css/style.css     # Styling
│   ├── js/main.js        # Frontend logic
│   └── img/              # Static images
├── models/               # ML model files
├── data/                 # Data files
└── temp/                 # Temporary uploads
```

## Contributing

This is a final project submission. For improvements or modifications, please fork the repository.

## License

MIT License - See LICENSE file for details.