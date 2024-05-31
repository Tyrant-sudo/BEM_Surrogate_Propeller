#=##############################################################################
# DESCRIPTION
    Postprocessing DJI 9443 simulations and comparison to experimental data
=###############################################################################

import FLOWUnsteady as uns
import PyPlot as plt
import CSV
import DataFrames: DataFrame
import Printf: @printf
import PyPlot: @L_str

uns.formatpyplot()

println("\nPostprocessing...\n")

# Path to where simulations are stored
sims_path = "/home/sh/WCY/auto_propeller/resource5/3_validation"

# Simulations to plot
sims_to_plot = [ # run_name, style, color, alpha, label
                ("rotorhover_MidHigh_initial", "-", "dodgerblue", 1.0, "rVPM - high fidelity")
              ]
nrotors = 1                 # Number of rotors
coli    = 1                 # Column to plot (1==CT, 2==CQ, 3==eta)
nrevs_to_average = 1        # Number of revolutions to average

nsteps_per_rev = Dict()
CTmean = Dict()
CTstd = Dict()

for (run_name, stl, clr, alpha, lbl) in sims_to_plot

    simdata = CSV.read(joinpath(sims_path, run_name, run_name*"_convergence1.csv"), DataFrame)


    # Calculate nsteps_per_rev
    nsteps_per_rev[run_name] = ceil(Int, 360 / (simdata[2, 1] - simdata[1, 1]))

    # Calculate mean CT and std dev
    roti = 1                # Rotor to average
    nsteps_to_average = nrevs_to_average*nsteps_per_rev[run_name]
    data_to_average = simdata[end-nsteps_to_average:end, 3 + (roti-1)*4 + 1+coli]

    CTmean[run_name] = uns.mean(data_to_average)
    CTstd[run_name] = sqrt(uns.mean((data_to_average .- CTmean[run_name]).^2))

end
              


################################################################################
#   Blade loading
################################################################################

rotor_axis = [-1.0, 0.0, 0.0]       # Rotor centerline axis
R          = 0.12                   # (m) rotor radius

# Generate statistics (mean and deviation of loading)
for (run_name, _, _, _, _) in sims_to_plot

    read_path = joinpath(sims_path, run_name)
    save_path = read_path*"-statistics"

    # Process outputs between revolutions 8 and 9
    nums      = range(8, 9; length = nsteps_per_rev[run_name]+1)
    nums      = ceil.(Int, nums * nsteps_per_rev[run_name])

    uns.postprocess_statistics(read_path, save_path, nums;
                                        cyl_axial_dir = rotor_axis,
                                        prompt = false)
end

# Start plotting



fig = plt.figure(figsize=[7*1.5, 5] * 2/3)
ax = fig.gca()



using CSV
using DataFrames

# 假设 sims_to_plot, sims_path, rotor_axis 等已经定义

# 初始化一个DataFrame来存储数据
df = DataFrame(Rs_R = Float64[], Np = Float64[])

for (run_name, stl, clr, alpha, lbl) in sims_to_plot
    read_path = joinpath(sims_path, run_name*"-statistics")

    (rs, Gamma, Np, Tp) = uns.postprocess_bladeloading(read_path;
                                                       O = zeros(3),
                                                       rotor_axis = rotor_axis,
                                                       filename = run_name*"_Rotor_Blade1_vlm-statistics.vtk",
                                                       fieldsuff = "-mean")

    # 添加数据到DataFrame
    append!(df, DataFrame(Rs_R = rs/R, Np = Np))
    ax.plot(rs/R, Np, stl; alpha=alpha, label=lbl, color=clr, linewidth=2.0)
end

read_path = joinpath(sims_path, sims_to_plot[1][1]*"-statistics")
# 保存DataFrame到CSV文件
CSV.write(read_path*"/blade_loading.csv", df)


exit()
# Format plot
xlims, dx = [[0, 1], 0.2]
ylims, dy = [[-1, 20], 5]

ax.set_xlim(xlims)
ax.set_xticks(xlims[1]:dx:xlims[2])
ax.set_xlabel(L"Radial position $r/R$")

ax.set_ylim(ylims)
ax.set_yticks(0:dy:ylims[2])
ax.set_ylabel(L"Loading ($\mathrm{N/m}$)")

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=false, fontsize=10)

ax.spines["right"].set_visible(false)
ax.spines["top"].set_visible(false)

fig.tight_layout()

# Save plot
fig.savefig("dji9443-loadingcomparison.png", dpi=300, transparent=true)
