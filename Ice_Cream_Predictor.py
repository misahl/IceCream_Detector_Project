import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('ice_cream_data.csv')

# Step 2: Separate inputs and outputs
X = data[['Temperature']]  # Input (Temperature)
y = data['WantsIceCream']   # Output (Yes=1, No=0)

# Step 3: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Test the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Predict new temperature
new_temp = float(input("Enter today's temperature (°C): "))
prediction = model.predict([[new_temp]])

if prediction[0] == 1:
    print("Prediction: Yes, the person will want ice cream! 🍦")
else:
    print("Prediction: No, the person won't want ice cream.")

# (Optional) Step 7: Show a simple graph
plt.scatter(X, y, color='blue')
plt.xlabel('Temperature (°C)')
plt.ylabel('Wants Ice Cream (0 = No, 1 = Yes)')
plt.title('Temperature vs Ice Cream Desire')
plt.show()
