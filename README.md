# PulsePredict: AI-Powered Cardiovascular Risk Assessment

**PulsePredict** is a state-of-the-art machine learning application designed to assess cardiovascular disease (CVD) risk and empower users with personalized health education. Built with a focus on impact and clarity, the system provides instant risk estimations, personalized lifestyle recommendations, and comprehensive medical insights.

## 🚀 Key Features

### 1. **Interactive Prediction Wizard**
A 7-step guided input process that collects:
- **Identity & Demographics**: Name, Age, Gender.
- **Physical Metrics**: Vitals like Systolic/Diastolic Blood Pressure, Height, and Weight.
- **Clinical Data**: Cholesterol and Glucose levels.
- **Lifestyle Factors**: Activity levels, Smoking, and Alcohol consumption.

### 2. **Instant Risk & Heart Age Analysis**
- **ML-Powered Estimation**: Uses a trained Logistic Regression model to calculate the probability of cardiovascular disease.
- **Heart Age**: A comparative metric that simplifies clinical data by telling users how their heart is performing relative to their chronological age.

### 3. **Personalized Recommendations with Impact Metrics**
- **Dynamic Guidance**: Tailored advice based on individual risk factors (e.g., "Quit Smoking", "Manage BMI").
- **Quantified Reduction**: Exclusive feature showing exactly how much each specific change (e.g., getting active) could **reduce your individual risk percentage**.

### 4. **Disease Intelligence Page**
- **Permanent Damage Alerts**: Highlights the irreversible consequences of CVD like heart scarring and organ failure.
- **Warning Symptoms**: Detailed breakdown of common and specific warning signs.
- **Consequences of Neglect**: A serious look at the outcomes of ignoring early symptoms.

### 5. **Comprehensive PDF Reports**
- Professional, clinical-style reports generated instantly via **ReportLab**.
- Includes risk scores, personal metrics, and a full list of recommendations for offline reference.

### 6. **Premium UI/UX**
- Custom-built Streamlit interface using **Modern Card-UI styling**.
- Responsive, accessible, and designed for a professional healthcare feel.

## 🛠️ Technical Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Customized with CSS/HTML injection)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Logistic Regression, Standard Scaling)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Reporting**: [ReportLab](https://www.reportlab.com/) (PDF Generation/Platypus)
- **Model Serialization**: Joblib

## 📖 Installation & Usage

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/your-username/PulsePredict.git](https://github.com/your-username/PulsePredict.git)
   cd PulsePredict
