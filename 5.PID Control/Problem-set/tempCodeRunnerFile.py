 in range(5)]
ORIGINAL += [[4, i] for i in range(1, 5)]
ORIGINAL += [[i, 4] for i in range(5, 10)]
ORIGINAL += [[9, i] for i in range(5, 10)]
ORIGINAL += [[i, 9] for i in range(10, 15)]
ORIGINAL += [[14, i] for i in range(10, 15)]
ORIGINAL += [[i, 14] for i in range(15, 20)]
ORIGINAL += [[19, i] for i in range(15, 20)]
ORIGINAL = np.array(ORIGINAL, dtype=np.float64)