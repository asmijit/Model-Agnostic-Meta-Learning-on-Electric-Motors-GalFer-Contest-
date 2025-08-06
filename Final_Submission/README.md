#  Gallileo Ferraris Contest 2025  
## Final Submission – Team MLotors

Welcome to the official submission repository of **Team MLotors** for the **Galfer Contest 2025**.  
This repository includes all necessary scripts and model files used for the **interpolation** and **extrapolation** tasks in the competition.

---

## Repository Structure

```
├── interpolation.py            # Script for the interpolation task  
├── extrapolation.py            # Script for the extrapolation task  
├── interpolation_models/      # Directory containing interpolation model weights  
├── extrapolation_models/      # Directory containing extrapolation model weights  
├── requirements.txt            # Dependencies required to run the scripts  
```

---

## Installation

To set up your environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Code

### Interpolation  
Call the `interpolation` function with the following **8 input parameters**:

```python
interpolation(d_alpha, h_c, r, w_t, l_t, w_o, dxIB, gamma)
```

### Extrapolation  
Call the `extrapolation` function with the following **12 input parameters**:

```python
extrapolation(d_alpha, h_c, r, w_t, l_t, w_o, dxIB, gamma, Rs, l_c, n_p, n_spp)
```

---


