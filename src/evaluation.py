from sklearn.metrics import mean_absolute_error

def calculate_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)
if __name__ == "__main__":
    print("This is a utility file. Use from main.py")