import pandas as pd

# Data for each model
data = {
    "Model": [
        "EfficientNetV2B0",
        "InceptionV3",
        "MobileNetV2",
        "MobileNetV3Large",
        "MobileNetV3Small",
        "ResNet50V2"
    ],
    "Total Parameters": [
        "6,070,470",
        "22,044,566",
        "2,409,142",
        "3,109,750",
        "1,007,206",
        "23,806,582"
    ],
    "Trainable Parameters": [
        "6,009,862",
        "22,010,134",
        "2,375,030",
        "3,085,350",
        "995,094",
        "23,761,142"
    ],
    "Non-Trainable Parameters": [
        "60,608",
        "34,432",
        "34,112",
        "24,400",
        "12,112",
        "45,440"
    ],
    "Initial Epoch Accuracy": [
        "0.4821",
        "0.4590",
        "0.2754",
        "0.0932",
        "0.0736",
        "0.3924"
    ],
    "Final Validation Accuracy": [
        "0.9026",
        "0.8648",
        "0.9026",
        "0.7596",
        "0.0932",
        "0.8732"
    ],
    "Training Time (minutes)": [
        "148.96",
        "219.43",
        "195.40",
        "253.54",
        "169.21",
        "288.46"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv('model_summary.csv', index=False)

# display the DataFrame in a Jupyter notebook
# df
