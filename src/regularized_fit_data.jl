"""
    regularized_fit_data

Returns a dictionary `D` with data for the regularized data fitting 
example on page 329 of VMLS.  The items in the dictionary are:
* `D["xtrain"]`: vector of length 10 
* `D["ytrain"]`: vector of length 10 
* `D["xtest"]`: vector of length 20 
* `D["ytest"]`: vector of length 20.

"""
function regularized_fit_data()
    X = [ 
    1.06830e-01  1.82422e+00;
    2.55780e-01  8.13549e-01;
    2.71120e-01  9.24060e-01;
    3.02990e-01  4.20670e-01;
    5.02040e-01  4.46178e-01;
    6.58450e-01 -3.73407e-02;
    6.84230e-01 -2.10935e-01;
    7.02590e-01 -1.03327e-01;
    7.34060e-01  3.32097e-01;
    9.40350e-01  2.29278e+00 
    ];
    xtrain = X[:,1];
    ytrain = X[:,2];
    Xtest = [
    3.96674e-01 -1.05772e+00;
    7.77517e-01  8.79117e-01;
    1.18400e-01  1.98136e+00;
    2.23266e-01  8.67012e-01;
    9.01463e-01  2.13650e+00;
    3.58033e-01 -7.01948e-01;
    2.60402e-01  9.41469e-01;
    8.04281e-01  1.49755e+00;
    6.31664e-01  5.50205e-01;
    1.49704e-01  1.34245e+00;
    5.51333e-01  1.21445e+00;
    6.63999e-01  4.49111e-03;
    1.64948e-01  9.57535e-01;
    6.51698e-01  7.78640e-02;
    1.23026e-01  1.73558e+00;
    3.37066e-01 -3.25244e-01;
    8.32080e-02  2.56555e+00;
    2.04422e-01  1.10081e+00;
    9.78000e-01  2.70644e+00;
    4.03676e-01 -1.10049e+00 
    ];
    xtest = Xtest[:,1];
    ytest = Xtest[:,2];
    return Dict([ ("xtrain", xtrain), ("ytrain", ytrain), 
        ("xtest", xtest), ("ytest", ytest) ])
end



