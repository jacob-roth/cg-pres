include("dynamics.jl")
include("auglag-cg.jl")
include("sgMAM.jl")
include("plotting.jl")

##
## example
##

p = Dict()
p[:d]  = 2
p[:N]  = 50
p[:A]  = I
p[:h]  = 0.05
p[:β]  = 10.0
p[:c]  = 0.5
p[:x0] = [-1.0, 0.0]
p[:α]  = 1e-2
p[:ϕ]  = ϕ
p[:max_iter] = 250
p[:tol] = 1e-4
p[:μ]  = 10.0
p[:λ]  = 10.0
p[:ϵ]  = 1e-4
u0 = ones(p[:d], p[:N]) ./ norm(ones(p[:d], p[:N]))^1.5

## solve
p[:J]    = J_al
p[:dJdu] = dJ_aldu
ustar = auglag(u0, p)
xstar = integrate(ustar, p)

## confirm with geometric MAM
p[:N]  = 101
p[:a]     = I
p[:inv_a] = I
p[:x1]    = xstar[:,end]
add_H!(p)
xstar_sgmam = sgMAM(p)

sgmam_label = "sgmam"
auglag_label = label="auglag-cg: " * raw"$\Phi(x_N)$" * " = $(@sprintf("%.2e", ϕ(xstar[:,end], p)))"

fig = figure(figsize=(10,10))
gs  = gspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
ax  = subplot(get(gs, (0, 0)))
ax[:cla]()
ax[:plot](xstar_sgmam[1,:], xstar_sgmam[2,:], label=sgmam_label, marker="o", alpha=0.5, markersize=15, color=colors[2])
ax[:plot](xstar[1,:], xstar[2,:], label=auglag_label, marker="X", markersize=15, color=colors[1])
ax[:legend](loc="lower right")
ax[:grid](true, linestyle="--")
ax[:legend]()
ax[:set_xlabel](raw"$x_1$")
ax[:set_ylabel](raw"$x_2$")
fig[:savefig]("optimal-control.png", dpi=600)