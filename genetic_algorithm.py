# Reese Zimmermann
# CS 4013 - hw2_genetic.py

# Genetic algorithm: Core Idea: Mimic biological evolution. Maintain a population of solutions, let the best ones
# "reproduce" to create new solutions, and occasionally introduce random "mutations.
# Use fitness-proportionate (roulette) selection, one-point crossover, and bit-flip mutation.
# Parameters: population size, number of generations, mutation rate, random seed, elitism count

import csv, math, random

# Read the dataset and convert categorical fields to 0/1
    # Gender: M=1, F=0
    # CarOwner: Y=1, N=0
    # PropertyOwner: Y=1, N=0
    # #Children, WorkPhone, Email_ID: integers
def load_dataset(path):
    X, y = [], []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gender = 1 if row['Gender'].strip().upper() == 'M' else 0
            car = 1 if row['CarOwner'].strip().upper() == 'Y' else 0
            prop = 1 if row['PropertyOwner'].strip().upper() == 'Y' else 0
            children = int(row['#Children'])
            workphone = int(row['WorkPhone'])
            email = int(row['Email_ID'])
            yi = int(row['CreditApprove'])
            X.append([gender, car, prop, children, workphone, email])
            y.append(yi)
    return X, y # Returns: X: list of feature lists, y: list of target values
# Predict the output for a single data row using weights w {-1, +1}
# Linear model f(x) = sum_j w_j * x_j
# row: list of feature values for a single data point
def predict_row(w, row):
    return sum(wj * xj for wj, xj in zip(w, row))

def error_mse(w, X, y):
    # compute mean squared error over dataset
    total = 0.0
    for row, yi in zip(X, y):
        fx = predict_row(w, row)
        total += (fx - yi) ** 2
    return total / len(X)
# er(w): = (1/n) * sum_i (f(x_i) - y_i)^2

# Fitness for genetic algorithm:
#   fitness = exp(-er(w))  (higher is better; always positive)
#   Also return er(w) because we log/compare errors explicitly.
def fitness(w, X, y):
    er = error_mse(w, X, y)
    return math.exp(-er), er

# Fitness-proportionate (roulette-wheel) selection.
#   pop_with_fit: list of tuples (fit, er, w)
#   Returns one selected chromosome w.
def roulette_select(pop_with_fit):
    # if all fitness values are zero (or negative), select randomly
    total_fit = sum(f for f, _er, _w in pop_with_fit)
    if total_fit <= 0:
        return pop_with_fit[random.randrange(len(pop_with_fit))][2]
    # spin the wheel in [0, total_fit)]
    r = random.random() * total_fit
    acc = 0.0
    for f, _er, w in pop_with_fit:
        acc += f
        if acc >= r:
            return w
    # should not reach here, but return last chromosome just in case
    # (due to possible floating-point precision issues)
    return pop_with_fit[-1][2]

# Single fixed-point crossover at the middle (after index 2 for length 6).
# Returns two children: [p1[:3] + p2[3:], p2[:3] + p1[3:]]
def crossover_mid(p1, p2):
    cut = len(p1)//2
    return p1[:cut]+p2[cut:], p2[:cut]+p1[cut:]

# Bit-flip mutation with given mutation rate (probability of flipping each gene).
def mutate(w, rate=0.1):
    return [(-g if random.random() < rate else g) for g in w]

def genetic_algorithm(X, y, pop_size=20, generations=50, mutation_rate=0.1, seed=2025, elitism_k=1):
    """
    Genetic algorithm to minimize er(w):
      - Chromosome representation: list of six values in {-1, +1}
      - Initialization: random ±1
      - Fitness: exp(-er(w))
      - Selection: roulette wheel
      - Crossover: fixed midpoint
      - Mutation: sign flip with probability 'mutation_rate'
      - Elitism: keep 'elitism_k' best chromosomes each generation

    Returns:
      best_w: best solution seen across all generations (best-so-far)
      best_er: its error
      curve: monotone non-increasing list recording best-so-far error per generation
    """
    random.seed(seed)   
    # Initialize population
    pop = [[random.choice([-1,1]) for _ in range(6)] for __ in range(pop_size)]
    best_w, best_er = None, float('inf')    #track best-so-far
    curve = []  
    
    for _ in range(generations):
        pop_with_fit = []
        # Evaluate population and update best-so-far
        for w in pop:
            f, er = fitness(w, X, y)
            pop_with_fit.append((f, er, w))
            if er < best_er:
                best_er, best_w = er, w[:]

    # Log best-so-far error, not just gen-best (updated)
        gen_best_er = min(er for _, er, _ in pop_with_fit)
        if not curve:
            curve.append(gen_best_er)
        else:
            curve.append(min(curve[-1], gen_best_er))

        # Elitism + reproduction -- *changed*:
        # Reproduction (elitism + crossover + mutation) to form new population
        # keep top-k elites (lowest er)
        elites = [w for _f,_er,w in sorted(pop_with_fit, key=lambda t: t[1])[:elitism_k]]
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1 = roulette_select(pop_with_fit)
            p2 = roulette_select(pop_with_fit)
            c1, c2 = crossover_mid(p1, p2)
            new_pop.extend([mutate(c1, mutation_rate), mutate(c2, mutation_rate)])
        pop = new_pop[:pop_size]

    
    return best_w, best_er, curve

def fmt_commas(w):
    parts = ["{:2d}".format(int(x)) for x in w]
    return "[ " + ", ".join(parts) + " ]"

# Add this helper (generated by ChatGPT) to save history to CSV
def save_history_csv(path, hist):
    with open(path,'w',newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['Generation','Error']) 
        for i,e in enumerate(hist, start=1):
            wr.writerow([i, e])

def main():
    # Load dataset
    X, y = load_dataset('CreditCard.csv')
    # Run genetic algorithm
    w, err, curve = genetic_algorithm(X, y, pop_size=8, generations=50, mutation_rate=0.2, seed=2025, elitism_k=0)
    
    # Print results
    print('Optimal w found by genetic algorithm:', fmt_commas(w))
    print('Optimal error er(w):', err)
    print("genetic_errors.csv")
    
    # Write CSV file
    with open('genetic_errors.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f) # create a CSV writer object
        writer.writerow(['Generation',' Error_er(w)']) # write header
        for gen, e in enumerate(curve, start=1): # write each generation and its error
            writer.writerow([gen, f"{e:.6f}"])
        

if __name__ == '__main__':
    main()
    """
    AI Assistance & Documentation
    
     Changes made with help from ChatGPT (OpenAI) and GitHub Copilot.
     - Comments and docstrings added throughout for clarity.
        - Helper function save_history_csv() added to save convergence history to CSV.
        - Code formatting and variable naming improved for readability.
    - Clear understanding of genetic algorithm components:
        - Chromosome representation, initialization, fitness function, selection, crossover, mutation, elitism.
        - Reproduction (elitism + crossover + mutation) to form new population.
        - Termination criteria (e.g., max generations, convergence).
    * Used for a breakdown of the algorithm and code structure *

        How to run:
        - Ensure CreditCard.csv is in the same folder as this script.
        - Run this file directly (e.g. python hw2_genetic.py).
        - It will output the optimal weights, error, and create a CSV file with the error per generation.
    """
