# Air Quality Prediction - Poprad, Slovakia

An air quality prediction system designed to forecast AQI (Air Quality Index) for Poprad, Slovakia using historical air‑quality and meteorological data. The model is deployed as a live demo on Hugging Face Spaces.

## 🌍 Demo

🔗 **Live Demo**: [Poprad AQI Prediction on Hugging Face](https://huggingface.co/spaces/martin-rajniak/poprad_aqi_prediction)

## 📋 Key Features
- AQI prediction for Poprad region
- Interactive web application for easy access
- Multiple machine learning algorithms comparison
- Visualization of air quality trends and predictions
- Historical data analysis and forecasting

## 🗂️ Project Structure

```
Air-Quality-Prediction/
├── .github/workflows      # GitHub Workflows for automation
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── scripts/               # Entry points for Docker, GitHub and Hopsworks
├── src/                   # Source code
│   ├── data               # Data loading and preprocessing 
│   ├── hopsworks          # Hopsworks' Feature Store, Model Registry and Deployment
│   ├── model              # Model training and evaluation
├── .dockerignore          # Specify code that should not be copied to Docker Image
├── Dockerfile             # Definition of Docker image used for automation
├── requirements.txt       # Python dependencies
```

## 🔬 Methodology

### Data Collection
The project utilizes various environmental parameters that influence air quality:
- **Particulate Matter**: PM2.5, PM10 concentrations
- **Gaseous Pollutants**: NO2, SO2, CO levels
- **Meteorological Data**: Temperature, humidity, wind speed, atmospheric pressure
- **Temporal Features**: Hour, day, month, season

### Machine Learning Models
The project implements and compares several ML algorithms:
- **Gradient Boosting**: Advanced ensemble technique (XGBoost/LightGBM)
- **Long Short-Term Memory (LSTM)**: For time series prediction

### Inference
- **Prediction Window of 3 days** - based on last 3 days (inputs), model predicts next 3 days (targets)
- **Recursive Forecasting** - a multi-step time series forecasting method where a model trained for one-step-ahead prediction is used iteratively to generate forecasts for multiple steps into the future

### Evaluation
- Using single metric for model comparison - **The Willmott index** - it gives credit for correlation but heavily penalizes systematic errors that would make the forecasts unreliable for air quality management.
- With recursive forecasting error is cumulative - cannot improve Day 3 prediction without improving Day 1 - so it is enough to **evaluate only last day's results for simplicity**.
- Instead **evaluate each output separately** (PM2.5, PM10, NO2, SO2, CO) as they tend to differ significantly.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MartinRajniak/Air-Quality-Prediction.git
cd Air-Quality-Prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Build Docker Image
```bash
./scripts/docker_build.sh
```

#### Run Any Python Automation Script
```bash
./scripts/docker_run.sh scripts/fetch_data.py
```

## 📊 Data Sources
The project integrates data from multiple sources:
- **Slovak Hydrometeorological Institute (SHMÚ)**: Official air quality monitoring stations
- **European Environment Agency**: Regional air quality data
- **OpenWeatherMap API**: Meteorological parameters
- **Local monitoring stations**: Real-time sensor data

## 🎯 Model Performance
The Willmott index (Index of Agreement) based on last day's predictions. Higher the better.

| Model | PM2.5 | PM10 | NO2 | SO2 | CO |
|-------|----------|------|-----|------|------|
| XGBoost | 0.3716 | 0.3599 | 0.4726 | 0.7681 | 0.211 |
| LSTM | 0.2846 | 0.4156 | 0.4476 | 0.6584 | 0.2417 |


## 🌟 Features

### Web Application Features
- **AQI Predictions**: Generate instant AQI prediction for Poprad, SVK for next 3 days.

### Technical Features
- **Automated Data Pipeline**: Regular data collection and model updates
- **Model Versioning**: Track and compare different model versions

## 🗺️ About Poprad

Poprad is located in northern Slovakia, near the High Tatras mountains. The city's air quality is influenced by:
- **Geographic Location**: Valley position affects air circulation
- **Industrial Activity**: Local manufacturing and transportation
- **Seasonal Variations**: Winter heating and summer tourism patterns
- **Weather Patterns**: Mountain weather systems and temperature inversions

## 📈 Results and Insights

### Key Findings
- **Seasonal Patterns**: Higher AQI values during winter months due to heating
- **Weather Dependencies**: Strong correlation with wind speed and atmospheric pressure
- **Daily Cycles**: Peak pollution during morning and evening rush hours
- **Feature Importance**: PM2.5 and meteorological conditions are primary predictors

### Visualizations
The project includes various visualizations:
- Time series plots of air quality trends
- Correlation heatmaps between variables
- Feature importance rankings
- Prediction vs actual value comparisons
- Geographic distribution of air quality

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:
```
AQI_TOKEN=your_aqi_api_key
HOPSWORKS_AQI_TOKEN=your_hopsworks_api_key
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Martin Rajniak**
- GitHub: [@MartinRajniak](https://github.com/MartinRajniak)

## 🙏 Acknowledgments

- Slovak Hydrometeorological Institute for providing air quality data
- [World Air Quality Index Project](https://waqi.info/) for providing the API 
- European Environment Agency for regional environmental data
- Open-source community for machine learning libraries
- Contributors and collaborators

## 🔄 Updates and Changelog

### Version 1.0.0 (Current)
- Initial release with basic prediction functionality
- Web application deployment on Hugging Face
- Multiple ML algorithm implementation

### Planned Updates
- [ ] Mobile application development
- [ ] Advanced deep learning models
- [ ] Integration with more data sources
- [ ] Multi-city prediction capability
- [ ] API rate limiting and authentication

---

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainer directly.