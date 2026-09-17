# Reese Zimmermann 
# CS 4013 - hw2_local.py
# hw2_local.py 
import csv, math, random
# csv:    read/write CSV files (dataset + convergence history)
# math:   reserved here (not strictly needed, but fine to import)
# random: randomized initialization (and reproducibility with a fixed seed)

DATA_PATH = "CreditCard.csv"  # default dataset location

def load_data(path=DATA_PATH):
    """
    Load the credit dataset and convert relevant fields to numeric features.

    Feature encoding (as in the assignment handout):
      Gender: 'M' -> 1, otherwise -> 0
      CarOwner: 'Y' -> 1, otherwise -> 0
      PropertyOwner: 'Y' -> 1, otherwise -> 0
      #Children, WorkPhone, Email_ID: parsed as integers
      CreditApprove: target y (0/1), parsed as int

    Returns:
      X: list of rows, each row is [gender, car, prop, children, work, email] (floats)
      y: list of targets (floats)
    """
    X, y = [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Categorical to numeric (simple 0/1 encodings)
            gender = 1 if row['Gender'] == 'M' else 0
            car = 1 if row['CarOwner'] == 'Y' else 0
            prop = 1 if row['PropertyOwner'] == 'Y' else 0

            # Numeric fields -> ints
            children = int(row['#Children'])
            work = int(row['WorkPhone'])
            email = int(row['Email_ID'])

            # Target (0/1)
            target = int(row['CreditApprove'])

            # Store as floats (matches how we do arithmetic later)
            X.append([float(gender), float(car), float(prop),
                      float(children), float(work), float(email)])
            y.append(float(target))
    return X, y

def dot(a, b):
    """Compute dot product sum_j a[j] * b[j] for two equal-length lists."""
    return sum(x * y for x, y in zip(a, b))

def er(w, X, y):
    """
    Mean squared error over the dataset for weight vector w:
      er(w) = (1/n) * sum_i ( dot(w, X[i]) - y[i] )^2
    """
    n = len(X)
    total = 0.0
    for i in range(n):
        fx = dot(w, X[i])        # model prediction for row i: f(x_i) = w · x_i
        total += (fx - y[i])**2  # squared residual
    return total / n             # average over all rows

def neighbors(w):
    """
    Generate all Hamming-1 neighbors of w in {-1, +1}^d
    (flip the sign of exactly one coordinate).

    Example: w = [1, -1, 1] -> neighbors:
      [-1, -1,  1], [1, 1, 1], [1, -1, -1]
    """
    out = []
    for i in range(len(w)):
        w2 = w[:]         # copy current weights
        w2[i] = -w2[i]    # flip one coordinate
        out.append(w2)
    return out

def hill_climb(X, y, w0=None, max_rounds=200, seed=42):
    """
    Deterministic hill-climbing (first-improvement via exhaustive neighbor scan):

    - Representation: w is a list of floats, each initialized to ±1.0
    - Objective: minimize er(w)
    - Move set: flip any single coordinate (neighbors(w))
    - Acceptance: move only if it strictly improves error (with tiny tolerance)
    - Termination: when no single flip improves error or max_rounds reached

    Returns:
      w:    best weight vector found
      hist: monotone non-increasing list of best-so-far errors per accepted move
    """
    random.seed(seed)           # reproducibility
    d = len(X[0])               # number of features (should be 6)
    # Initialize: either use provided w0 or random ±1.0 for each gene
    w = [random.choice([-1.0, 1.0]) for _ in range(d)] if w0 is None else [float(v) for v in w0]

    hist = []                   # error trace after each successful improvement
    cur = er(w, X, y)           # current error
    hist.append(cur)            # log initial error
    rounds = 0

    # Try up to max_rounds improvement steps
    while rounds < max_rounds:
        rounds += 1
        best_err, best_w = cur, w[:]  # track the best neighbor this round

        # Examine all single-bit flips (neighbors)
        for w2 in neighbors(w):
            e2 = er(w2, X, y)
            # strict improvement with small numeric tolerance
            if e2 < best_err - 1e-12:
                best_err, best_w = e2, w2

        # If we found an improving neighbor, move there and record error
        if best_err + 1e-12 < cur:
            w, cur = best_w, best_err
            hist.append(cur)
        else:
            # No improvement among neighbors -> reached a local minimum
            break

    return w, hist

def save_history_csv(path, hist):
    """
    Save the convergence history to CSV with columns:
      Round, Error
    Note: rounds start at 0 for the initial solution (before any move).
    """
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Round', 'Error'])
        for i, e in enumerate(hist):
            wr.writerow([i, e])

if __name__ == "__main__":
    # 1) Load data
    X, y = load_data()

    # 2) Optional fixed start (all -1). If you want random starts, set w0=None.
    w0 = [-1, -1, -1, -1, -1, -1]

    # 3) Run hill climbing with at most 200 accepted-improvement steps
    w, h = hill_climb(X, y, w0=w0, max_rounds=200, seed=42)

    # 4) Save convergence to CSV (monotone best-so-far by construction)
    save_history_csv("hill_climb_results.csv", h)

    # 5) Print final result summary
    print("Optimal w found by hill climbing:", [int(v) for v in w])
    print("Optimal error er(w):", h[-1])
    print("hill_climb_results.csv")

    """ 
    How to run: 
    - The program will execute when you run this file directly with the CreditCard.csv in folder
    - It will output the optimal weights, error, and create a CSV file with the figure
    
    AI Assistance & Documentation
1. Had chatGPT generate a full detailed doc for the implementation process, what the assignmnent is doing,
 and how the code should be structured.
    - This was to clarify my own understanding of the assignment and ensure I was on the right track.
    - The doc included a full breakdown of the functions and their purposes.
2. Explanations for hill_climb function
   - The hill_climb function implements a deterministic hill-climbing algorithm to minimize the error er(w) for a given weight vector w.
3. Explanations for CSV format - Matplotlib not able to be used on my system
   - The save_history_csv function saves the convergence history of the hill-climbing algorithm to a CSV file.
   - It includes columns for the round number and the corresponding error value.
   - The CSV file can be used to analyze the optimization process and visualize the error reduction over time.
    """
