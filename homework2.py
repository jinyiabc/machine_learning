import numpy as np
value_min = 0
value_rand = 0
value_first = 0
for j in range(100000):
    
    coin_qty = 1000
    min = 11
    min_idx = 10001
    eps = np.finfo(float).eps
    data = np.random.uniform(low=0.0, high=1.0 + eps, size=(coin_qty, 10))
    data_conversion = data >= 0.5
    
    # Count heads for respective coins after 10 independent flips.
    coin_count = np.zeros(coin_qty)
    for idx, value in enumerate(coin_count):
        coin_count[idx] = np.count_nonzero(data_conversion[idx])
        value = coin_count[idx]
        if value < min:
            min = value
            min_idx = idx
    # Select a min head frequency coin.
    coin_min = np.array([min_idx, min])  
    
    # select a random coin with head frequency.     
    rand_idx = np.random.randint(low=0, high=1000) 
    coin_rand = np.array([rand_idx, coin_count[rand_idx]])
    
    # Select the 1st coin.
    coin_first = np.array([0, coin_count[0]])
    #print(coin_rand)
    #print(coin_min)    
    #print(coin_first)    
    value_min = value_min + coin_min[1]
    value_rand = value_rand + coin_rand[1]
    value_first = value_first + coin_first[1]
print(value_min/100000)    
print(value_rand/100000)  
print(value_first/100000)  