using LinearAlgebra
using ForwardDiff

##
## objective function
##

function J(u, p)
    u = reshape(u, p[:d], p[:N])
    xx = integrate(u, p)
    return sum(u.^2) + ϕ(xx[:,end], p)
end
function dJdu(u, p)
    return ForwardDiff.gradient(uu -> J(uu, p), u)
end

##
## augmented lagrangian
##

function J_al(u, λ, μ, p)
    u = reshape(u, p[:d], p[:N])
    xx = integrate(u, p)
    return sum(u.^2) - λ * ϕ(xx[:,end], p) + (μ/2.0) * ϕ(xx[:,end], p)^2
end
function J_al(u, p)
    return J_al(u, p[:λ], p[:μ], p)
end
function dJ_aldu(u, p)
    return ForwardDiff.gradient(uu -> J_al(uu, p), u)
end

##
## optimization
##

function verbose_1(u, g, p, i)
    println("┌──────────────")
    println("│ iter  : $i")
    println("│ ||g|| : $(norm(g))")
    println("│ ||u|| : $(norm(u))")
    println("│ J(u)  : $(J(u, p))")
    println("│ c(u)  : $(ϕ(integrate(u, p)[:, end], p))")
    println("└──────────────")
end
function verbose_2(u, g, p, i)
    println(">>  │ iter  : $i")
    println(">>  │ ||g|| : $(norm(g))")
    println(">>  │ ||u|| : $(norm(u))")
    println(">>  │ J(u)  : $(J(u, p))")
end
function verbose_3(u, g, p, i)
    println(">>  >>  │ iter  : $i")
    println(">>  >>  │ ||g|| : $(norm(g))")
    println(">>  >>  │ ||u|| : $(norm(u))")
    println(">>  >>  │ J(u)  : $(J(u, p))")
end

function btls(f, ∇f, x, p, d; c=0.4, rho=0.9, max_iter=1000)
    if (c <= 0) || (rho <= 0) || (c >= 1) || (rho >= 1)
        error("must choose c and rho in (0,1)")
    end
    ## check if `d` is a descent direction
    fx, gx = f(x, p), ∇f(x, p)
    deriv = sum(gx .* d)
    if deriv >= 0
        @warn("direction is not a descent direction\n")
        return -1e-5, 0
    ## else perform backtracking linesearch
    end
    alpha = 1.0
    iter  = 1
    while true
        ## armijo condition
        fx_hat = f(x .+ alpha .* d, p)
        thresh = fx .+ (c .* alpha) .* sum(gx .* d)
        if fx_hat <= thresh
            break
        else
            alpha *= rho
        end
        iter += 1
        if iter >= max_iter
            @warn("max iters of $max_iter exceeded in btls\n")
            break
        end
    end
    return alpha, iter
end

function ncg(u0, p)
    max_iter = p[:max_iter]
    tol  = p[:tol]
    j    = p[:J]
    djdu = p[:dJdu]
    u    = deepcopy(u0)
    gu0  = ones(size(u0))
    gu1  = ones(size(u0))
    d    = -dJdu(u0, p)
    k    = 1
    res  = 500
    while (norm(gu1) >= tol) && (k <= max_iter)
        verbose_2(u, d, p, k)
        alpha, iters = btls(j, djdu, u, p, d)
        u  .+= alpha .* d
        gu0 .= d
        gu1 .= djdu(u, p)
        beta = sum(gu1 .* (gu1 .- gu0)) ./ sum(gu0.^2)
        if mod(k, res) == 0
            d   .= -gu1
        else
            d   .= -gu1 .+ beta .* d
        end
        k   += 1
    end
    return u, d
end

function auglag(u0, p)
    u = u0
    ϕ = p[:ϕ]
    η = 1.0/(p[:μ]^0.1)
    con_tol = p[:tol]
    g_tol   = p[:tol]
    k = 1
    while true
        u,g = ncg(u, p)
        x   = integrate(u, p)
        nc  = norm(ϕ(x[:,end], p))
        ng  = norm(g)
        verbose_1(u, g, p, k)
        if (nc <= η)
            if (nc <= con_tol) && (ng <= g_tol)
                break
            else
                p[:λ] -= p[:μ] * ϕ(x[:,end], p)
                η  = η/p[:μ]^0.9
            end
        else
            p[:μ] *= 100.0
            η  = 1/p[:μ]^0.1
        end
        k += 1
        if k >= p[:max_iter]
            @warn("maximum AugLag iterations reached")
            break
        end
    end
    return u
end
