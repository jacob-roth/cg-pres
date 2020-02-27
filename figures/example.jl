using ForwardDiff
using LinearAlgebra
include("plotting.jl")

##
## problem
##

const A1 = [4. 0.;
            0. 1.]

const A2 = [4. 1.;
            1. 1.]

const b  = [1., 1.]
const v1 = eigvecs(A1)
const v2 = eigvecs(A2)
const x0 = [1.0, 5.0]
# A2\b + eigvecs(A2)[:,1] #+ [0.025, 0.0]

##
## functions
##

function f(x, p)
    return 0.5 * x' * (p[:A] * x) - p[:b]' * x
end

function ∇f(x, p)
    """ Ax - b """
    return ForwardDiff.gradient(xx -> f(xx, p), x)
end

function gd(x0, p, steepest=false)
    """ gradient descent / steepest descent """
    x  = deepcopy(x0)
    xh = [deepcopy(x)]
    while true
        g  = ∇f(x, p)
        if steepest
            x -= ((g'*g) / (g'*p[:A]*g)) .* g
        else
            x -= p[:α] .* g
        end
        push!(xh, deepcopy(x))
        if sum(g.^2) <= p[:tol]
            break
        end
    end
    return hcat(xh...)
end

function cd(x0, p, dirn=0)
    """ coordinate descent """
    x  = deepcopy(x0)
    xh = [deepcopy(x)]
    n  = length(x)
    _f = x -> f(x,p)
    _g = x -> ∇f(x,p)
    g  = zeros(n)
    while true
        if dirn==0
            idx = collect(1:n)
        else
            idx = collect(n:-1:1)
        end
        for k in idx
            gk = zeros(n)
            g = -∇f(x, p)
            gk[k] = g[k]
            alpha, _, _, _, _, _ = opt.line_search(_f, _g, x, gk)
            x += alpha*gk
            push!(xh, deepcopy(x))
        end
        if sum(g.^2) <= p[:tol]
            break
        end
    end
    return hcat(xh...)
end

function cg(x0, p)
    x    = deepcopy(x0)
    xh   = [deepcopy(x)]
    gx   = +∇f(x,p)  # Ax-b
    gTg  = gx'*gx
    d    = -gx       # -(Ax-b)
    while true
        Ad    = p[:A]*d
        alpha = gTg / (d'*Ad)
        x   .+= alpha .* d; push!(xh, deepcopy(x))
        gx   .= gx + alpha * Ad
        _gTg  = gx'*gx
        beta  = _gTg / (gTg)
        d    .= -gx .+ beta .* d
        gTg   = _gTg
        if gTg <= p[:tol]
            break
        end
    end
    return hcat(xh...)
end

function draw_background(ax, p)
    x1 = collect(range(-2, stop=2, length=100))
    x2 = collect(range(0, stop=6, length=100))
    xx = meshgrid(x1, x2)
    zz = zeros(100, 100)
    for i in 1:100
        for j in 1:100
            ff = f([xx.x[i,j], xx.y[i,j]], p)
            zz[i,j] = ff
        end
    end
    ax[:contourf](x1, x2, zz, sort(collect(range(f(p[:x0], p), stop=f(p[:xx], p), length=10))), cmap=get_cmap("bone"), zorder=1, alpha=0.75)
    return ax
end
function add_points(ax, p)
    ax[:scatter](p[:x0][1], p[:x0][2], marker="o", color="blue", s=100, label=raw"$x_0$", zorder=10, alpha=0.75)
    ax[:scatter](p[:xx][1], p[:xx][2], marker="*", color="fuchsia",  s=250, label=raw"$x^{\star}$", zorder=10, alpha=0.75)
    return ax
end

##
## A1-A2
##
fig = figure(figsize=(10,10))
gs = gspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
AA = [A1, A2]
ll = [raw"$A_1$", raw"$A_2$"]
mm = ["<",">", "^", "v"]
p = Dict()
p[:tol] = 1e-3
p[:α]   = 0.1
p[:hist]= true
p[:x0]  = x0

## contours
for i in eachindex(AA)
    p[:A]   = AA[i]
    p[:b]   = b
    p[:xx]  = p[:A]\p[:b]
    ax = subplot(get(gs, (slc(0,2), i-1)))
    ax[:cla]()
    ax = draw_background(ax, p)
    ax = add_points(ax, p)
    ax[:legend](loc="lower right")
end
fig[:tight_layout]()
fig[:savefig]("figures/contours.png", dpi=600)

## gd
for i in eachindex(AA)
    p[:A]   = AA[i]
    p[:b]   = b
    p[:xx]  = p[:A]\p[:b]
    xh_sd = gd(p[:x0], p, true)
    ax = subplot(get(gs, (slc(0,2), i-1)))
    ax[:plot](xh_sd[1,:], xh_sd[2,:], color="lightcoral",  label="SD: " * ll[i], marker=mm[1], linewidth=2, alpha=0.75)
    ax[:legend](loc="lower right")
end
fig[:tight_layout]()
fig[:savefig]("figures/sd.png", dpi=600)

## gd
for i in eachindex(AA)
    p[:A]   = AA[i]
    p[:b]   = b
    p[:xx]  = p[:A]\p[:b]
    xh_gd = gd(p[:x0], p, false)
    ax = subplot(get(gs, (slc(0,2), i-1)))
    # ax = draw_background(ax, p)
    ax[:plot](xh_gd[1,:], xh_gd[2,:], color="lightsalmon",    label="GD: " * ll[i], marker=mm[2], linewidth=2, alpha=0.75)
    ax[:legend](loc="lower right")
end
fig[:tight_layout]()
fig[:savefig]("figures/sd_gd.png", dpi=600)

## cd
for i in eachindex(AA)
    p[:A]   = AA[i]
    p[:b]   = b
    p[:xx]  = p[:A]\p[:b]
    xh_cd = cd(p[:x0], p)
    ax = subplot(get(gs, (slc(0,2), i-1)))
    # ax = draw_background(ax, p)
    ax[:plot](xh_cd[1,:], xh_cd[2,:], color="cornflowerblue", label="CD: " * ll[i], marker=mm[3], linewidth=2, alpha=0.75)
    ax[:legend](loc="lower right")
end
fig[:tight_layout]()
fig[:savefig]("figures/sd_gd_cd.png", dpi=600)

## cg
for i in eachindex(AA)
    p[:A]   = AA[i]
    p[:b]   = b
    p[:xx]  = p[:A]\p[:b]
    xh_cg = cg(p[:x0], p)
    ax = subplot(get(gs, (slc(0,2), i-1)))
    # ax = draw_background(ax, p)
    ax[:plot](xh_cg[1,:], xh_cg[2,:], color="lightsteelblue", label="CG: " * ll[i], marker=mm[4], linewidth=2, alpha=0.75)
    ax[:legend](loc="lower right")
end
fig[:tight_layout]()
fig[:savefig]("figures/sd_gd_cd_cg.png", dpi=600)

## combined
for i in eachindex(AA)
    p[:A]   = AA[i]
    p[:b]   = b
    p[:xx]  = p[:A]\p[:b]
    xh_sd = gd(p[:x0], p, true)
    xh_gd = gd(p[:x0], p, false)
    xh_cd = cd(p[:x0], p)
    xh_cg = cg(p[:x0], p)
    ax = subplot(get(gs, (slc(0,2), i-1)))
    ax[:cla]()
    ax = draw_background(ax, p)
    ax = add_points(ax, p)
    ax[:plot](xh_sd[1,:], xh_sd[2,:], color="lavender",  label="SD: " * ll[i], marker=mm[mod1(i,4)], alpha=0.5, linewidth=2)
    ax[:plot](xh_gd[1,:], xh_gd[2,:], color="violet",    label="GD: " * ll[i], marker=mm[mod1(i+1,4)], alpha=0.25, linewidth=2)
    ax[:plot](xh_cd[1,:], xh_cd[2,:], color="mistyrose", label="CD: " * ll[i], marker=mm[mod1(i+2,4)], linewidth=2)
    ax[:plot](xh_cg[1,:], xh_cg[2,:], color="peachpuff", label="CG: " * ll[i], marker=mm[mod1(i+3,4)], linewidth=2)
    ax[:legend]()
end
fig[:savefig]("figures/sd_gd_cd_cg.png", dpi=600)