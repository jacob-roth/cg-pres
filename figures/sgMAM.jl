using Interpolations

##
## helpers
##

function lapply_mat_cols(M, x)
    v = zeros(eltype(x), size(x[:,1]))
    z = zeros(eltype(x), size(x))
    for i in 1:size(x,2)
        @inbounds @views mul!(v, M, x[:,i])
        @inbounds @views z[:,i] = v
    end
    return z
end
function apply_fun_cols(f, x)
    v = zeros(eltype(x), size(f(x[:,1])))
    z = zeros(eltype(x), size(v,1), size(x,2))
    for i in 1:size(x,2)
        @inbounds @views z[:,i] = f(x[:,i])
    end
    return z
end
function apply_fun_cols(f, x, y)
    size(x) == size(y) || throw(BoundsError)
    v = zeros(eltype(x), size(f(x[:,1], y[:,1])))
    z = zeros(eltype(x), size(v,1), size(x,2))
    for i in 1:size(x,2)
        @inbounds @views z[:,i] = f(x[:,i], y[:,i])
    end
    return z
end
function get_prime(z)
    return hcat(zeros(size(z[:,1])), 0.5*(z[:,3:end] .- z[:,1:end-2]), zeros(size(z[:,1])))
end
function get_s(N)
    return collect(range(0.0, stop=1.0, length=N))
end
function init_path(x0, x1, s)
    xx = hcat( (x1[1] - x0[1]) .* s .+ x0[1],
               (x1[2] - x0[2]) .* s .+ x0[2] .+ 0.1 .* s .* (1.0 .- s)
             )'
    return convert(Array{Float64,2}, xx)
end

##
## multiplicative noise
##

function get_θstar(x, xdot, μ, p)
    λ = get_λ(μ)
    return lapply_mat_cols(p[:inv_a], (xdot .* λ) .- apply_fun_cols(xx -> p[:b](xx,p), x))
end
function get_μ(x, xdot, p)
    bx         = apply_fun_cols(xx -> p[:b](xx, p), x)
    inv_a_bx   = lapply_mat_cols(p[:inv_a], bx)
    inv_a_xdot = lapply_mat_cols(p[:inv_a], xdot)
    μ          = sqrt.(sum((xdot .* inv_a_xdot), dims=1)) ./ sqrt.(sum((bx .* inv_a_bx), dims=1))
    μ[1]   = 0.0
    μ[end] = 0.0
    return μ
end
function get_λ(μ)
    λ      = 1.0 ./ μ
    λ[1]   = 0.0
    λ[end] = 0.0
    return λ
end

##
## main function
##

function sgMAM(p)
    x0  = p[:x0]
    x1  = p[:x1]
    s   = get_s(p[:N])
    α   = zeros(p[:N])
    x   = init_path(x0, x1, s)
    tol = p[:tol]
    max_iter = 20_000

    k = 1
    while true
        ## compute gradient flow
        xdot     = get_prime(x)
        xdotdot  = get_prime(xdot)
        μ        = get_μ(x, xdot, p)
        λ        = get_λ(μ)
        θstar    = get_θstar(x, xdot, μ, p)
        θstardot = get_prime(θstar)
        H_x      = apply_fun_cols((xx,θθ) -> p[:H_x](xx, θθ, p), x, θstar)
        d        = p[:ϵ] .* (λ .* θstardot .+ H_x)
        d[:,1]     .= 0.0
        d[:, end]  .= 0.0
        x      .+= d

        ## interpolate
        α .= vec(hcat(0.0, cumsum(sqrt.(sum((x[:,2:end]-x[:,1:end-1]).^2, dims=1)),dims=2)))
        α .= α/α[end]
        itp_d = [interpolate((α,), x[dd,:], Gridded(Linear())) for dd in 1:2]
        x .= convert(Array{Float64,2}, hcat([d(s) for d in itp_d]...)')

        ## check convergence
        if (norm(d) <= tol) || k >= max_iter
            break
        end
        k += 1
        println(norm(d))
    end
    return x
end