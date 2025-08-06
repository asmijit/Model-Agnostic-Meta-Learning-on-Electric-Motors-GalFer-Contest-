import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBRegressor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import joblib


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # Output is now 6
        )
    
    def forward(self, x):
        return self.seq(x)

class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

def add_physics_features(Rs, r, l_c, n_p, n_spp, gamma, wt, lt, wo, dxIB, h_c):
    airgap_area = np.pi * (Rs**2 - r**2)
    conductor_length = 2 * (wt + lt)
    winding_volume = n_p * n_spp * conductor_length * wo
    slot_fill_factor = winding_volume / (airgap_area * l_c)
    Rs_lc = Rs*l_c
    gamma_np = gamma*n_p
    dxIB_hc = dxIB*h_c
    
    return np.array([airgap_area, conductor_length, winding_volume, slot_fill_factor,
        Rs_lc, gamma_np, dxIB_hc])

def extrapolation(d_alpha,h_c,r,w_t,l_t,w_o,dxIB,gamma,Rs,l_c,n_p,n_spp):

    scaler_x = joblib.load('extrapolation_models/scaler_x.gz')
    scaler_y = joblib.load('extrapolation_models/scaler_y.gz')

       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    X=np.array([d_alpha,h_c,r,w_t,l_t,w_o,dxIB,gamma,Rs,l_c,n_p,n_spp])
    X = X.reshape(1,12)
    
    X = scaler_x.transform(X)
    X = torch.tensor(X,dtype=torch.float32)
    print(X.dtype)
    
    model = Net().to(device)
    model.load_state_dict(torch.load("extrapolation_models/nn_6", weights_only=False,map_location=torch.device('cpu')))

    xgb_models=[]
    for i in range(6):
        loaded_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        loaded_model.load_model(f'extrapolation_models/xgb_{i}.json')
        xgb_models.append(loaded_model)
    
    
    model.eval()
    with torch.no_grad():
        y_test_nn_pred_scaled = model(X.to(device)).cpu()
    # print(X.unsqueeze(X,0))
    corrections=[]
    for i, xgb in enumerate(xgb_models):
        corr = xgb.predict(X.view(1, -1).cpu())
        corrections.append(corr[0])


    corrections=np.array(corrections)
    y_test_final_scaled = y_test_nn_pred_scaled + corrections
    y_pred_6 = scaler_y.inverse_transform(y_test_final_scaled.reshape(1, -1))


    # DANGER ABOVE
    
    
    model_TR = FNN(input_dim=(19)).to(device)
    model_TR.load_state_dict(torch.load("extrapolation_models/TR", weights_only=False,map_location=torch.device('cpu')))

    scaler = joblib.load('extrapolation_models/scaler.gz')
    
    data_point = add_physics_features(Rs, r, l_c, n_p, n_spp, gamma, w_t, l_t, w_o, dxIB, h_c)

    features = ['d_alpha', 'h_c', 'r', 'w_t', 'l_t', 'w_o', 'dxIB', 'gamma', 'Rs', 'l_c', 'n_p', 'n_spp',
        'airgap_area', 'conductor_length', 'winding_volume', 'slot_fill_factor',
        'Rs_lc', 'gamma_np', 'dxIB_hc']
    target = 'TR'

    X_test = np.array([d_alpha,h_c,r,w_t,l_t,w_o,dxIB,gamma,Rs,l_c,n_p,n_spp])
    X_test = np.append(X_test, data_point)
    
    X_test = scaler.transform(X_test.reshape(1, -1))
    
    X_test = torch.tensor(X_test, dtype = torch.float32)
    
    model_TR.eval()
    with torch.no_grad():
        y_pred_log = model_TR(X_test).flatten().numpy()
        y_pred = np.expm1(y_pred_log)

    y_preds = np.insert(y_pred_6, 1, y_pred)

    
    return y_preds.reshape(-1)
    
    
