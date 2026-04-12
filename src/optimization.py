import numpy as np

# =========================
# EOQ Calculation
# =========================
def calculate_eoq(D, S, H):
    """
    D = Demand (units per period)
    S = Ordering cost
    H = Holding cost per unit
    """
    return np.sqrt((2 * D * S) / H)

# =========================
# Safety Stock
# =========================
def safety_stock(std_dev, lead_time, z=1.65):
    """
    std_dev = demand standard deviation
    lead_time = days
    z = service level factor (1.65 ~ 95%)
    """
    return z * std_dev * np.sqrt(lead_time)