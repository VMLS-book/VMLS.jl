"""
A library of functions used in the Julia Companion to 
the book **Vectors, Matrices, and Least Squares**.
"""
module VMLS

using LinearAlgebra
using SparseArrays
using DelimitedFiles
export avg, rms, stdev, ang, correl_coef, eye, speye, kmeans, gram_schmidt
export diagonal, spdiagonal, vandermonde, toeplitz, linspace
export confusion_matrix, row_argmax, one_hot, mols_solve, cls_solve
export levenberg_marquardt, aug_lag_method
export house_sales_data, population_data, petroleum_consumption_data
export vehicle_miles_data, temperature_data, iris_data, ozone_data
export regularized_fit_data, portfolio_data, lq_estimation_data
export orth_dist_reg_data

"""
    avg(x)

Returns the average of the elements of the vector or matrix `x`.
"""
avg(x) = sum(x) / length(x)


"""
    rms(x)

Returns the RMS value of the elements of the vector or matrix `x`.
"""
rms(x) = norm(x) / sqrt(length(x))


"""
    stdev(x)

Returns the standard deviation of the elements of the vector or matrix `x`.
"""
stdev(x) = rms(x .- avg(x))


"""
    ang(x,y)

Returns the angle in radians between non-zero vectors `x` and `y`.
"""
ang(x,y) = acos(x'*y  / (norm(x) * norm(y)))


"""
    correl_coef(x,y)

Returns the correlation coefficient between non-constant vectors `x` and `y`.
"""
function correl_coef(x,y) 
    xdm = x .- avg(x)
    ydm = y .- avg(y)
    return xdm'*ydm / (norm(xdm) * norm(ydm)) 
end


"""
    eye(n)

Returns an `n` times `n` identity matrix.
"""
eye(n) = Matrix(1.0I, n, n)


"""
    speye(n)

Returns an `n` times `n` sparse identity matrix.
"""
speye(n) = sparse(1.0I, n, n)


"""
    kmeans(X, k; maxiters = 100, tol = 1e-5)

Applies the k-means algorithm for `k` clusters to the vectors stored in `X`.

The argument `X` is a one-dimensional array of N n-vectors or an `n` times `N` matrix.

`kmeans` returns a tuple with two elements: `assignment` is an array of N 
integers between 1 and k with the cluster assignment for the N data points.
`reps` is an array of k n-vectors with the k cluster representatives. 
"""
function kmeans(X, k; maxiters = 100, tol = 1e-5)
    if ndims(X) == 2
        X = [X[:,i] for i in 1:size(X,2)]
    end;
    N = length(X)
    n = length(X[1])
    distances = zeros(N)  
    reps = [zeros(n) for j=1:k]  
    assignment = [ rand(1:k) for i in 1:N ]
    Jprevious = Inf  
    for iter = 1:maxiters
        for j = 1:k
            group = [i for i=1:N if assignment[i] == j]             
            reps[j] = sum(X[group]) / length(group);
        end;
        for i = 1:N
            (distances[i], assignment[i]) = 
                findmin([norm(X[i] - reps[j]) for j = 1:k]) 
        end;
        J = norm(distances)^2 /  N
        println("Iteration ", iter, ": Jclust = ", J, ".")
        if iter > 1 && abs(J - Jprevious) < tol * J  
            return assignment, reps
        end
        Jprevious = J
    end
end


"""
    gram_schmidt(a; tol = 1e-10)

Applies the Gram-Schmidt algorithm to the vectors stored in the array `a` 
and returns the result as an array of (orthonormal) vectors.  Issues a
warning if the original vectors are linearly dependent.
"""
function gram_schmidt(a; tol = 1e-10)
    q = []
    for i = 1:length(a)
        qtilde = a[i]
        for j = 1:i-1
            qtilde -= (q[j]'*a[i]) * q[j]
        end
        if norm(qtilde) < tol
            println("Vectors are linearly dependent.")
            return q
        end
        push!(q, qtilde/norm(qtilde)) 
    end;
    return q
end

"""
    diagonal(x)

Returns a diagonal matrix with the entries of the vector `x` on its diagonal. 
"""
diagonal(x) = diagm(0 => x)


"""
    spdiagonal(x)

Returns a sparse diagonal matrix with the entries of the vector `x` on its diagonal. 
"""
spdiagonal(x) = spdiagm(0 => x)


"""
    vandermonde(t, n)

Returns the Vandermonde matrix with `n` columns and i-th column `t`^(i-1).
"""
vandermonde(t,n) = hcat( [t.^i for i = 0:n-1]... )


"""
    toeplitz(a, n)

Returns the Toeplitz matrix with `n` columns and the vector `a` in the
leading positions of the first column.
"""
function toeplitz(a,n)
    m = length(a)
    T = zeros(n+m-1,n)
    for i=1:m
        T[i : n+m : end] .= a[i]
    end
    return T
end


"""
    linspace(a,b,n)

Returns a vector with `n` equally spaced numbers between `a` and `b`.
"""
linspace(a,b,n) = Vector(range(a, stop = b, length = n))


"""
    confusion_matrix(y, yhat, K=2)

Returns the confusion matrix for a data vector `y` and the vector of predictions
`yhat`.  If `K` is 2, the vectors are Boolean.  If `K` is greater than 2, they 
contain integers from 1,...,K.
"""
function confusion_matrix(y, yhat, K=2)
    if K==2
        tp = sum( (y .== true) .& (yhat .== true) ) 
        fn = sum( (y .== true) .& (yhat .== false) ) 
        fp = sum( (y .== false) .& (yhat .== true) ) 
        tn = sum( (y .== false) .& (yhat .== false) ) 
        return [tp fn; fp tn]
    else
        C = zeros(K,K)
        for i = 1:K  for j=1:K
            C[i,j] = sum( (y .== i) .& (yhat .== j) )
        end end
        return C
    end;
end;


"""
    row_argmax(A)

Returns a `size(A,1)` vector with as i-th element the column index of the 
largest element in row i of `A`.
"""
row_argmax(A) = [ argmax(A[i,:]) for i = 1:size(A,1) ]


"""
    one_hot(x, K)

Returns the one-hot encoding of the vector `x`.
"""
function one_hot(y, K)
    n = length(y)
    Y = zeros(n, K)
    for j in 1:K
       Y[findall(y .== j), j] .= 1.0
    end
    return Y
end


"""
    mols_solve(As, bs, lambdas)

Returns the solution of the multi-objective least squares problem 
with coefficient matrices in the array `As`, right-hand side vectors in the
array `bs`, and weights in the array `lambdas`.
"""
function mols_solve(As,bs,lambdas)
    k = length(lambdas);
    Atil = vcat([sqrt(lambdas[i])*As[i] for i=1:k]...)
    btil = vcat([sqrt(lambdas[i])*bs[i] for i=1:k]...)
    return Atil \ btil
end


"""
   cls_solve(A, b, C, d)

Returns the solution of the constrained least squares problem with coefficient 
matrices `A` and `C`, and right-hand side vectors or matrices `b` and `d`.
"""
function cls_solve(A,b,C,d)
    m, n = size(A)
    p, n = size(C)
    Q, R = qr([A; C])
    Q = Matrix(Q)
    Q1 = Q[1:m,:]
    Q2 = Q[m+1:m+p,:]
    Qtil, Rtil = qr(Q2')
    Qtil = Matrix(Qtil)
    w = Rtil \ (2*Qtil'*Q1'*b - 2*(Rtil'\d))
    return R \ (Q1'*b - Q2'*w/2)
end


"""
    levenberg_marquardt(f, Df, x1, lambda1; kmax=500, tol=1e-6)

Applies the Levenberg-Marquardt algorithm to the function defined in `f` and
`Df`, with starting point `x1` and initial regularization parameter `lambda1`.
The function returns the final iterate `x` and a dictionary with the convergence 
history.
"""
function levenberg_marquardt(f, Df, x1, lambda1; kmax=100, tol=1e-6)
    n = length(x1)
    x = x1
    lambda = lambda1
    objectives = zeros(0,1)
    residuals = zeros(0,1)
    for k = 1:kmax
         fk = f(x)
         Dfk = Df(x)
         objectives = [objectives; norm(fk)^2]
         residuals = [residuals; norm(2*Dfk'*fk)]
         if norm(2*Dfk'*fk) < tol
             break  
         end;
         xt = x - [ Dfk; sqrt(lambda)*eye(n) ] \ [ fk; zeros(n) ]
         if norm(f(xt)) < norm(fk)
             lambda = 0.8*lambda
             x = xt
         else
             lambda = 2.0*lambda
         end
    end
    return x, Dict([ ("objectives", objectives), ("residuals", residuals) ])
end


"""
    aug_lag_method(f, Df, g, Dg, x1, lambda1; kmax=100, feas_tol=1e-4, oc_tol=1e-4)

Applies the augmented Lagrangian method to the constrained nonlinear least 
squares problem defined by `f`, `Df`, `g`, `Dg`, with starting point `x1`.
The subproblems are solved using the Levenberg-Marquardt method with 
initial regularization parameter `lambda1`.  Returns the final iterate `x`, 
multiplier `z`, and a dictionary with the convergence history.
"""
function aug_lag_method(f, Df, g, Dg, x1, lambda1;  kmax = 100, 
    feas_tol = 1e-4, oc_tol = 1e-4)
    x = x1
    z = zeros(length(g(x)))
    mu = 1.0
    feas_res = [norm(g(x))]
    oc_res = [norm(2*Df(x)'*f(x) + 2*mu*Dg(x)'*z)]
    lm_iters = zeros(Int64,0,1);
    mus = [mu]
    for k=1:kmax
        F(x) = [f(x); sqrt(mu)*(g(x) + z/(2*mu))] 
        DF(x) = [Df(x); sqrt(mu)*Dg(x)] 
        x, hist = levenberg_marquardt(F, DF, x, lambda1, tol = oc_tol)
        z = z + 2*mu*g(x)
        feas_res = [feas_res; norm(g(x))]
        oc_res = [oc_res; hist["residuals"][end]]
        lm_iters = [lm_iters; length(hist["residuals"])]
        if norm(g(x)) < feas_tol
            break
        end
        mu = (norm(g(x)) < 0.25*feas_res[end-1]) ? mu : 2*mu 
        mus = [mus; mu]
    end
    return x, z, Dict([ ("lm_iterations", lm_iters),  
         ("feas_res", feas_res), ("oc_res", oc_res), ("mus", mus) ])
end




"""
    petroleum_consumption_data()

Returns a 34-vector with the world annual petroleum consumption between 
1980 and 2013, in thousand barrels/day (discussed on page 252).
"""
petroleum_consumption_data() = [ 
    63122, 60953, 59551, 58785, 59795, 60083, 61819, 63107, 64978, 66090, 
    66541, 67186, 67396, 67619, 69006, 70258, 71880, 73597, 74274, 75975, 
    76928, 77732, 78457, 80089, 83063, 84558, 85566, 86724, 86046, 84972, 
    88157, 89105, 90340, 91195 ]

include("house_sales_data.jl")
include("population_data.jl")
include("vehicle_miles_data.jl")
include("temperature_data.jl")
include("iris_data.jl")
include("ozone_data.jl")
include("regularized_fit_data.jl")
include("portfolio_data.jl")
include("lq_estimation_data.jl")
include("orth_dist_reg_data.jl")

end # module
