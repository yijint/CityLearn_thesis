# Code is written to reproduce and adapt the MPC designed in 
# https://www.e3s-conferences.org/articles/e3sconf/abs/2023/33/e3sconf_iaqvec2023_04018/e3sconf_iaqvec2023_04018.html

import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib  # for saving the scaler

# Define LSTM-based forecasting model
class LSTMForecaster(nn.Module):
    def __init__(self, 
                 n_buildings, 
                 hidden_size=128, # to be finetuned
                 output_size=2, # forecasted load, forecasted solar power generation
                 num_layers=2 # to be finetuned
                ):
        """
        LSTM model to forecast future electricity load and solar power generation.
        """
        super(LSTMForecaster, self).__init__()
        input_size = (25 + 3*n_buildings) + 2 # data + sin-encoded timestamp, cos-encoded timestanp
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Predict next time step based on final time step

def preprocess_data(data, timestamps):
    """
    Standardizes data and encodes timestamps using both sine and cosine transformations.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Encode time with sin-cos transformation
    time_sin = np.sin((timestamps % 24) * (2 * np.pi / 24)).reshape(-1, 1)
    time_cos = np.cos((timestamps % 24) * (2 * np.pi / 24)).reshape(-1, 1)

    return np.hstack([data_scaled, time_sin, time_cos])  # Combine features

# Forecasting function with Recursive Multi-Step Prediction
def forecast(model, input_data, steps=12):
    """
    Forecasts future electricity load and solar power generation using an LSTM model.
    Performs recursive one-step-ahead prediction for 12 time steps.
    12 time steps is the control horizon defined in this paper. 
    """
    model.eval()
    predictions = []
    current_input = input_data.unsqueeze(0)  # Add batch dimension

    for _ in range(steps):
        with torch.no_grad():
            pred = model(current_input)  # Predict next step
        predictions.append(pred.numpy())  # Store the prediction
        
        # Append prediction to input and remove oldest time step
        pred_expanded = torch.cat((current_input[:, 1:, :], pred.unsqueeze(1)), dim=1)
        current_input = pred_expanded

    return np.array(predictions)  # Return full forecasted sequence

# Define MPC optimization function
def mpc_optimization(forecasted_load, forecasted_pv, 
                     price, carbon_intensity, battery):
    """
    Computes the optimal battery action over a 12-hour control horizon to minimize cost.
    
    Parameters:
    - forecasted_load (array): Forecasted electricity demand over the next 12 hours.
    - forecasted_pv (array): Forecasted solar generation over the next 12 hours.
    - price (array): Electricity price per kWh over the next 12 hours.
    - carbon_intensity (array): Carbon intensity per kWh over the next 12 hours.
    - battery (citylearn.energy_model.Battery): Battery model

    Returns:
    - First action in the optimal sequence.
    """
    from scipy.optimize import minimize
    
    def objective(actions):
        total_cost = 0
        q_cost, q_carbon = 2, 1  # Weighing factors (fine-tuned in the paper's test case 2)

        for t in range(len(actions)):
            battery.charge(actions[t]*battery.capacity)

            # Compute net load on the grid
            net_load = forecasted_load[t] - forecasted_pv[t] - actions[t]*battery.capacity # negative means selling to the grid
            cost = q_cost * price[t] * net_load + q_carbon * carbon_intensity[t] * net_load
            total_cost += cost

        return total_cost

    # Optimization using Powell's method 
    # (this is what the paper used, I am sticking with it since the efficiency curve
    # is piecewise, so a gradient-based approach like L-BFGS-B might not be appropriate
    initial_guess = -np.sign(price[:12] - np.mean(price[:12])) # Charge fully when price is low, discharge fully when price is high
    res = minimize(objective, initial_guess, bounds=[(-1.0, 1.0)] * 12, method='Powell')
    return res.x[0]  # Return first action in the optimal sequence

# # Battery model 
# # The power value can be negative, which means the battery is releasing energy. 
# class BatteryModel:
#     def __init__(self, capacity, nominal_power):
#         """
#         Battery model with dynamically calculated efficiency and power constraints.

#         Parameters:
#         - capacity (float): Maximum energy capacity of the battery in kWh.
#         - nominal_power (float): Maximum charging/discharging power in kW.
#         """
#         self.capacity = capacity # kWh
#         self.nominal_power = abs(nominal_power) # kW
#         self.soc = 0.5 * capacity  # Start battery at 50% SOC

#     def get_efficiency(self, power):
#         """Returns efficiency dynamically based on power level."""
#         # Use efficiency curve from charging data provided by Tesla users,
#         # illsutrated in Figure 6 from https://arxiv.org/pdf/2012.10504
#         pu_power_levels = np.array([0.0, 0.3, 0.7, 0.8, 1.0])  # fraction / per unit basis of nominal power
#         pu_power_efficiency_values = np.array([0.83, 0.83, 0.90, 0.90, 0.85])  # efficiency at each per unit basis power level
#         pu_power = abs(power) / self.nominal_power  # Convert to per-unit
#         return np.interp(pu_power, pu_power_levels, pu_power_efficiency_values)

#     def update(self, action):
#         """
#         Updates battery SOC based on charging/discharging action.

#         Parameters:
#         - action (float): Charging (-1 to 1), where:
#             - Positive values means charging
#             - Negative values means discharging
        
#         Returns:
#         - New SOC after applying action
#         """
#         # Compute actual battery power (limited by nominal power)
#         power = np.clip(action * self.nominal_power, -self.nominal_power, self.nominal_power)

#         # Dynamic power efficiency
#         efficiency = self.get_efficiency(power)
#         if power > 0:  # Power charged decreases
#             adjusted_power = power * efficiency
#         else:  # Power discharged increases
#             adjusted_power = power / efficiency 
            
#         # Update SOC and enforce limits
#         self.soc = np.clip(self.soc + adjusted_power, 0, self.capacity)
#         return self.soc

# Train LSTM
# Hyperparameters 
LOOKBACK = 8
PREDICT_HORIZON = 1 # one-step prediction
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_PATH = 'lstm_model.pt'
SCALER_PATH = 'scaler.gz'

# prepare training data 
def prepare_data(x_data, y_data):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    X, y = [], []
    for i in range(len(x_scaled) - LOOKBACK - PREDICT_HORIZON + 1):
        X.append(x_scaled[i:i+LOOKBACK])
        y.append(y_data[i+LOOKBACK])

    return np.array(X), np.array(y), scaler

# train and save LSTM model 
def train_and_save_lstm(X_train, y_train, n_buildings):
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMForecaster(n_buildings=n_buildings)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for batch_X, batch_y in loader:
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def save_scaler(scaler):
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

def load_model(n_buildings):
    model = LSTMForecaster(n_buildings=n_buildings)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def load_scaler():
    return joblib.load(SCALER_PATH)

    # # Simulate example data
    # time = np.arange(0, 500)
    # series = np.sin(2 * np.pi * time / 24) + 0.1 * np.random.randn(len(time))

    # X, y, scaler = prepare_data(series)
    # model = train_and_save_lstm(X, y)
    # save_scaler(scaler)

    # # Example recursive forecast
    # model = load_model()
    # scaler = load_scaler()

    # x_input = torch.tensor(X[-1:], dtype=torch.float32).unsqueeze(-1)
    # predictions = []
    # current_input = x_input.clone()

    # for _ in range(12):  # 12-hour forecast
    #     with torch.no_grad():
    #         pred = model(current_input)
    #     predictions.append(pred.item())

    #     # Update input sequence
    #     new_input = torch.cat([current_input[:, 1:, :], pred.unsqueeze(0).unsqueeze(-1)], dim=1)
    #     current_input = new_input

    # # Inverse scale
    # predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # print("Next 12-hour forecast:", predictions)
