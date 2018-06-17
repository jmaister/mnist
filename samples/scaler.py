def normalize(X):
    print(X[2])
    # regularize
    #X = (X - np.min(X)) / np.ptp(X)
    
    # train the normalization
    #scaler = RobustScaler(feature_range=(-1, 1))
    scaler = RobustScaler()
    scaler = scaler.fit(X)
    print('scaler', scaler)
    normalized = scaler.transform(X)

    print(normalized[2])
    
    return normalized