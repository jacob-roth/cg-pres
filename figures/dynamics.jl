
##
## dynamical system functions
##

function f(x::AbstractArray{T,1}, p) where T
    """ flow """
    return [
              x[1] - x[1]^3 - p[:β] * x[1] * x[2]^2  ;
              -(1.0 + x[1]^2) * x[2]
           ]
end
function dfdx(x::AbstractArray{T,1}, p) where T
    """ jacobian of flow """
    return [
              1.0 - 3.0 * x[1]^2 - p[:β] * x[2]^2      -2.0 * p[:β] * x[2] * x[1]  ;
             -2.0 * x[2] * x[1]                        -(1.0 + x[1]^2)
           ]
end
function F(x0::AbstractArray{T,1}, p) where T
    """ forward euler integrator """
    x  = x0
    x += p[:h] * f(x, p)
    return x
end
function F_prime(x::Array{T,1}, p) where T
    """ derivative of forward euler integrator """
    return ForwardDiff.jacobian(xx -> F(xx, p), x)
end

##
## terminal constraint functions
##

function ϕ(x::AbstractArray{T,1}, p) where T
    return sum( (x .- p[:x0]).^2 ) - p[:c]^2
end
function dϕdx(x::AbstractArray{T,1}, p) where T
    ## (-2.0 / (sum((x .- x0).^2) - c^2)) .* (x .- x0)
    return ForwardDiff.gradient(xx -> ϕ(xx, p), x)
end

##
## integration
##

function integrate(u, p)
    m, n = size(u)
    xx = zeros(eltype(u), m, n)
    xx[:,1] = p[:x0]
    for i in 1:(n-1)
        xx[:,i+1] .= F(xx[:,i], p) .+ u[:,i+1]
    end
    return xx
end

##
## Hamiltonian
##

function H(x, y, p)
    return sum(f(x, p) .* y) .+ 0.5sum(y.^2)
end
function H_xy(x, y, p)
    return dfdx(x, y, p)
end
function H_yy(x, y, p)
    return I
end
function inv_H_yy(x, y, p)
    return I
end
function H_x(x, y, p)
    return dfdx(x, p)' * y
end
function H_y(x, y, p)
    return f(x) .+ p[:a] * y
end
function add_H!(p::Dict)
    p[:b]        = f
    p[:dbdx]     = dfdx
    p[:H]        = H
    p[:H_x]      = H_x
    p[:H_y]      = H_y
    p[:H_xy]     = H_xy
    p[:H_yy]     = H_yy
    p[:inv_H_yy] = inv_H_yy
end