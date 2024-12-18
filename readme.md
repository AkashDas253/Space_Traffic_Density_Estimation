# Space Traffic Density Prediction

This project is a Streamlit-based application that predicts **space traffic density** based on location, date, time, and types of space objects. The app utilizes a pre-trained **RandomForestRegressor** model to generate predictions and a **LabelEncoder** to handle categorical data like locations.

## Features
- **User-Friendly Interface**: Enter data using dropdowns, date pickers, and checkboxes.
- **Custom Predictions**: Make predictions for traffic density in space for specific locations and times.
- **Reusable Model**: The app uses a pre-trained machine learning model and allows for consistent predictions.

---

## Setup Instructions

### Prerequisites
1. Python 3.9+ installed on your machine.
2. Dependencies listed in `requirements.txt` (see below).

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/space-traffic-prediction.git
   cd space-traffic-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `model` folder contains the following:
   - `RandomForestRegressor.joblib` (trained model)
   - `label_encoder.joblib` (trained LabelEncoder)

4. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## How to Use the App

1. **Input Fields**:
   - Select a **Location** from the dropdown menu.
   - Enter the **Year**, **Month**, and **Day** using the respective fields.
   - Select the relevant **Space Object Types** using the checkboxes.

2. **Click the 'Predict' Button**:
   - The app will display the **Predicted Traffic Density**.

3. **Example Input**:
   If you input the following:
   - **Location**: `Lagrange Point L2`
   - **Year**: `2024`
   - **Month**: `10`
   - **Day**: `21`
   - **Space Object Types**: `Space Station`

   The predicted output might be:
   ```plaintext
   Predicted Traffic Density: 25.8
   ```

---

## Example Input Data for Prediction

Here's a breakdown of the expected input:
| Field                          | Example Value          | Description                                 |
|--------------------------------|------------------------|---------------------------------------------|
| **Location**                   | `Lagrange Point L2`    | Dropdown of space locations                |
| **Year**                       | `2024`                | Year for prediction                        |
| **Month**                      | `10`                  | Month for prediction                       |
| **Day**                        | `21`                  | Day of the month for prediction            |
| **Object_Type_Asteroid Mining Ship** | `False`           | Checkbox for presence of object type       |
| **Object_Type_Manned Spacecraft**    | `False`           | Checkbox for presence of object type       |
| **Object_Type_Satellite**           | `False`           | Checkbox for presence of object type       |
| **Object_Type_Scientific Probe**     | `False`           | Checkbox for presence of object type       |
| **Object_Type_Space Debris**        | `False`           | Checkbox for presence of object type       |
| **Object_Type_Space Station**       | `True`            | Checkbox for presence of object type       |

---

## Folder Structure

```
space-traffic-prediction/
├── app.py                   # Streamlit application
├── model/                   # Folder for trained model and encoder
│   ├── RandomForestRegressor.joblib
│   ├── label_encoder.joblib
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## Dependencies

Install the dependencies using the provided `requirements.txt`:
```plaintext
streamlit==1.27.0
scikit-learn==1.3.1
pandas==2.1.2
joblib==1.3.2
```

---

## Contributing

Feel free to fork the repository and submit pull requests for new features or improvements.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
