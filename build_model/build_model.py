import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle



   

def train_crop_prediction_model(dataset_path):
    # Step 1: Load the data into a DataFrame
    df = pd.read_csv(dataset_path)
    print(df.head(10))

    # Step 2: Check for null values
    if df.isnull().values.any():
        print("Null values detected in the dataset. Please handle them before proceeding.")
        return None
    # drop col
    df = df.drop(columns=['Crop1', 'Crop2', 'Fertilizer', 'Fertilizer1','Fertilizer2','Link','Unnamed: 0'])


    
    df[['Crop', 'District_Name', 'Soil_color', 'Season']] = df[['Crop', 'District_Name', 'Soil_color', 'Season']].apply(lambda x: pd.factorize(x)[0])

    print(df.head(10))

    # Step 4: Prepare the data
    X = df.drop(columns=['Crop'])  # Features
    y = df['Crop']  # Target variable

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.head(10))

    # Step 6: Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 7: Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Step 8: Save the trained model
    with open('models\Random_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return accuracy

# Example usage:
dataset_path = 'data\df1.csv'  # Replace 'data\your_dataset.csv' with the actual path to your dataset file
accuracy = train_crop_prediction_model(dataset_path)
if accuracy:
    print("Model trained and saved successfully.")
    print("Accuracy:", accuracy)
