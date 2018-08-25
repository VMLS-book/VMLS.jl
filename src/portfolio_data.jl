"""
   portfolio_data()       

Returns a tuple `(R, Rtest)` with data for the portfolio optimization 
example in section 17.1.3.
`R` is a 2000 x 20 matrix with daily returns over a period of 2000 days.  
The first 19 columns are returns for 19 stocks; 
the last column is for a risk-free asset.
`Rtest` is a 500 x 20 matrix with daily returns over a different period 
of 500 days.
"""
function portfolio_data()
    pth = splitdir(pathof(VMLS))[1]
    prices = readdlm(joinpath(pth, "portfolio_data.txt"), comments = true);
    p_changes = prices[2:end,:] - prices[1:end-1,:];
    returns = p_changes ./ prices[1:end-1,:];
    R = returns[1:2000,:]; 
    Rtest = returns[2001:2500,:];
    return R, Rtest
end
