#=##############################################################################
# DESCRIPTION
    Simulation of a DJI 9443 rotor in hover (two-bladed rotor, 9.4 inches
    diameter).

    This example replicates the experiment described in Zawodny & Boyd (2016),
    "Acoustic Characterization and Prediction of Representative,
    Small-scale Rotary-wing Unmanned Aircraft System Components."

# AUTHORSHIP
  * Author          : Eduardo J. Alvarez (edoalvarez.com)
  * Email           : Edo.AlvarezR@gmail.com
  * Created         : Mar 2023
  * Last updated    : Mar 2023
  * License         : MIT
=###############################################################################

#=
Use the following parameters to obtain the desired fidelity

---- MID-LOW FIDELITY ---
n               = 20                        # Number of blade elements per blade
nsteps_per_rev  = 36                        # Time steps per revolution
p_per_step      = 4                         # Sheds per time step
sigma_rotor_surf= R/10                      # Rotor-on-VPM smoothing radius
vpm_integration = vpm.euler                 # VPM time integration scheme
vpm_SFS         = vpm.SFS_none              # VPM LES subfilter-scale model
shed_starting   = false                     # Whether to shed starting vortex
suppress_fountain    = true                 # Suppress hub fountain effect
sigmafactor_vpmonvlm = 1.0                  # Shrink particles by this factor when
                                            #  calculating VPM-on-VLM/Rotor induced velocities

---- MID-HIGH FIDELITY ---
n               = 50
nsteps_per_rev  = 72
p_per_step      = 2
sigma_rotor_surf= R/10
sigmafactor_vpmonvlm = 1.0
shed_starting   = false
suppress_fountain    = true
vpm_integration = vpm.rungekutta3
vpm_SFS         = vpm.SFS_none

---- HIGH FIDELITY -----
n               = 50
nsteps_per_rev  = 360
p_per_step      = 2
sigma_rotor_surf= R/80
sigmafactor_vpmonvlm = 5.5
shed_starting   = true
suppress_fountain    = false
vpm_integration = vpm.rungekutta3
vpm_SFS         = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
                                    alpha=0.999, maxC=1.0,
                                    clippings=[vpm.clipping_backscatter])
=#

import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm

run_name        =  "rotorhover_MidHigh_MaxFM"     # Name of this simulation
save_path       = run_name                  # Where to save this simulation
paraview        = true                      # Whether to visualize with Paraview

# Uncomment this to have the folder named after this file instead
# save_path     = String(split(@__FILE__, ".")[1])
# run_name      = "singlerotor"
# paraview      = false
# ----------------- GEOMETRY PARAMETERS ----------------------------------------

# Rotor geometry
rotor_file      = "DJI9443.csv"             # Rotor geometry
data_path       = uns.def_data_path         # Path to rotor database
pitch           = 0.0                       # (deg) collective pitch of blades
CW              = false                     # Clock-wise rotation
xfoil           = false                     # Whether to run XFOIL
read_polar      = vlm.ap.read_polar2        # What polar reader to use

# NOTE: If `xfoil=true`, XFOIL will be run to generate the airfoil polars used
#       by blade elements before starting the simulation. XFOIL is run
#       on the airfoil contours found in `rotor_file` at the corresponding
#       local Reynolds and Mach numbers along the blade.
#       Alternatively, the user can provide pre-computer airfoil polars using
#       `xfoil=false` and providing the polar files through `rotor_file`.
#       `read_polar` is the function that will be used to parse polar files. Use
#       `vlm.ap.read_polar` for files that are direct outputs of XFOIL (e.g., as
#       downloaded from www.airfoiltools.com). Use `vlm.ap.read_polar2` for CSV
#       files.

# Discretization
n               = 50                        # Number of blade elements per blade
r               = 1/10                      # Geometric expansion of elements

# NOTE: Here a geometric expansion of 1/10 means that the spacing between the
#       tip elements is 1/10 of the spacing between the hub elements. Refine the
#       discretization towards the blade tip like this in order to better
#       resolve the tip vortex.

# Read radius of this rotor and number of blades
R, B            = uns.read_rotor(rotor_file; data_path=data_path)[[1,3]]

# ----------------- SIMULATION PARAMETERS --------------------------------------

# Operating conditions
RPM             = 5400                      # RPM
J               = 0.0001                    # Advance ratio Vinf/(nD)
AOA             = 0                         # (deg) Angle of attack (incidence angle)

rho             = 1.071778                  # (kg/m^3) air density
mu              = 1.85508e-5                # (kg/ms) air dynamic viscosity
speedofsound    = 342.35                    # (m/s) speed of sound

# NOTE: For cases with zero freestream velocity, it is recommended that a
#       negligible small velocity is used instead of zero in order to avoid
#       potential numerical instabilities (hence, J here is negligible small
#       instead of zero)

magVinf         = J*RPM/60*(2*R)
Vinf(X, t)      = magVinf*[cos(AOA*pi/180), sin(AOA*pi/180), 0]  # (m/s) freestream velocity vector

ReD             = 2*pi*RPM/60*R * rho/mu * 2*R      # Diameter-based Reynolds number
Matip           = 2*pi*RPM/60 * R / speedofsound    # Tip Mach number

println("""
    RPM:    $(RPM)
    Vinf:   $(Vinf(zeros(3), 0)) m/s
    Matip:  $(round(Matip, digits=3))
    ReD:    $(round(ReD, digits=0))
""")

# ----------------- SOLVER PARAMETERS ------------------------------------------

# Aerodynamic solver
VehicleType     = uns.UVLMVehicle           # Unsteady solver
# VehicleType     = uns.QVLMVehicle         # Quasi-steady solver
const_solution  = VehicleType==uns.QVLMVehicle  # Whether to assume that the
                                                # solution is constant or not
# Time parameters
nrevs           = 10                        # Number of revolutions in simulation
nsteps_per_rev  = 72                        # Time steps per revolution
nsteps          = const_solution ? 2 : nrevs*nsteps_per_rev # Number of time steps
ttot            = nsteps/nsteps_per_rev / (RPM/60)       # (s) total simulation time

# VPM particle shedding
p_per_step      = 2                         # Sheds per time step
shed_starting   = false                     # Whether to shed starting vortex
shed_unsteady   = true                      # Whether to shed vorticity from unsteady loading
unsteady_shedcrit = 0.001                   # Shed unsteady loading whenever circulation
                                            #  fluctuates by more than this ratio
max_particles   = ((2*n+1)*B)*nsteps*p_per_step + 1 # Maximum number of particles

# Regularization
sigma_rotor_surf= R/10                      # Rotor-on-VPM smoothing radius
lambda_vpm      = 2.125                     # VPM core overlap
                                            # VPM smoothing radius
sigma_vpm_overwrite = lambda_vpm * 2*pi*R/(nsteps_per_rev*p_per_step)
sigmafactor_vpmonvlm= 1                     # Shrink particles by this factor when
                                            #  calculating VPM-on-VLM/Rotor induced velocities

# Rotor solver
vlm_rlx         = 0.5                       # VLM relaxation <-- this also applied to rotors
hubtiploss_correction = ((0.4, 5, 0.1, 0.05), (2, 1, 0.25, 0.05)) # Hub and tip correction

# VPM solver
# vpm_integration = vpm.euler                 # VPM temporal integration scheme
vpm_integration = vpm.rungekutta3

vpm_viscous     = vpm.Inviscid()            # VPM viscous diffusion scheme
# vpm_viscous   = vpm.CoreSpreading(-1, -1, vpm.zeta_fmm; beta=100.0, itmax=20, tol=1e-1)

vpm_SFS         = vpm.SFS_none              # VPM LES subfilter-scale model
# vpm_SFS       = vpm.SFS_Cd_twolevel_nobackscatter
# vpm_SFS       = vpm.SFS_Cd_threelevel_nobackscatter
# vpm_SFS       = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
#                                   alpha=0.999, maxC=1.0,
#                                   clippings=[vpm.clipping_backscatter])
# vpm_SFS       = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
#                                   alpha=0.999, rlxf=0.005, minC=0, maxC=1
#                                   clippings=[vpm.clipping_backscatter],
#                                   controls=[vpm.control_sigmasensor],
#                                   )

# NOTE: In most practical situations, open rotors operate at a Reynolds number
#       high enough that viscous diffusion in the wake is actually negligible.
#       Hence, it does not make much of a difference whether we run the
#       simulation with viscous diffusion enabled or not. On the other hand,
#       such high Reynolds numbers mean that the wake quickly becomes turbulent
#       and it is crucial to use a subfilter-scale (SFS) model to accurately
#       capture the turbulent decay of the wake (turbulent diffusion).

if VehicleType == uns.QVLMVehicle
    # Mute warnings regarding potential colinear vortex filaments. This is
    # needed since the quasi-steady solver will probe induced velocities at the
    # lifting line of the blade
    uns.vlm.VLMSolver._mute_warning(true)
end



# ----------------- WAKE TREATMENT ---------------------------------------------
# NOTE: It is known in the CFD community that rotor simulations with an
#       impulsive RPM start (*i.e.*, 0 to RPM in the first time step, as opposed
#       to gradually ramping up the RPM) leads to the hub "fountain effect",
#       with the root wake reversing the flow near the hub.
#       The fountain eventually goes away as the wake develops, but this happens
#       very slowly, which delays the convergence of the simulation to a steady
#       state. To accelerate convergence, here we define a wake treatment
#       procedure that suppresses the hub wake for the first three revolutions,
#       avoiding the fountain effect altogether.
#       This is especially helpful in low and mid-fidelity simulations.

suppress_fountain   = true                  # Toggle

# Supress wake shedding on blade elements inboard of this r/R radial station
no_shedding_Rthreshold = suppress_fountain ? 0.35 : 0.0

# Supress wake shedding for this many time steps
no_shedding_nstepsthreshold = 3*nsteps_per_rev

omit_shedding = []          # Index of blade elements to supress wake shedding

# Function to suppress or activate wake shedding


# ----------读取数据列-----------

using CSV
using DataFrames

# 定义一个解析函数，将字符串转换为浮点数数组
function parse_float_array(str::String)
    # 去除不需要的字符
    clean_str = replace(str, '[' => "", ']' => "", '\n' => "")
    # 基于空格将字符串分割为数组，并转换为浮点数
    return parse.(Float64, split(clean_str))
end


df = CSV.read("Opmized_ChordPitch.csv", DataFrame)
println(names(df))

# 应用解析函数到每个需要的列
df[!, :"r/R(chord)"] = parse_float_array.(df[!, :"r/R(chord)"])
df[!, :"c/R"] = parse_float_array.(df[!, :"c/R"])
df[!, :"r/R(pitch)"] = parse_float_array.(df[!, :"r/R(pitch)"])
df[!, :"twist (deg)"] = parse_float_array.(df[!, :"twist (deg)"])


Rtip, Rhub, B, blade_file = uns.read_rotor(rotor_file; data_path=data_path)
(chorddist, pitchdist,
sweepdist, heightdist,
airfoil_files, spl_k,
spl_s) = uns.read_blade(blade_file; data_path=data_path)

N = 4
col_chord     = df[2, :"r/R(chord)"]
col_pitch     = df[2, :"r/R(pitch)"]

# 假设 col1 和 col2 是等长的数组
length_chord = length(col_chord)
length_pitch = length(col_pitch)
Dist_chorddist = Array{Float64}(undef, N, length_chord, 2)
Dist_pitchdist = Array{Float64}(undef, N, length_pitch, 2)

for i in 2:2
    colchord1 = df[i, :"r/R(chord)"]
    colchord2 = df[i, :"c/R"]
    Dist_chorddist[i, :, 1] = colchord1
    Dist_chorddist[i, :, 2] = colchord2

    colpitch1 = df[i, :"r/R(pitch)"]
    colpitch2 = df[i, :"twist (deg)"]
    Dist_pitchdist[i, :, 1] = colpitch1 
    Dist_pitchdist[i, :, 2] = colpitch2
end


i = 1

for i in 2:2
local_chorddist = Dist_chorddist[i,:,:]
local_pitchdist = Dist_pitchdist[i,:,:]
conv_suff_path = "_convergence$(i).csv"

rotor = uns.generate_rotor(Rtip, Rhub, B, local_chorddist, local_pitchdist, sweepdist,
                            heightdist, airfoil_files;
                            spline_k=spl_k, spline_s=spl_s,
                            pitch=pitch,
                            n=n, CW=CW, blade_r=r,
                            altReD=[RPM, J, mu/rho],
                            xfoil=xfoil,
                            read_polar=read_polar,
                            data_path=data_path,
                            verbose=true,
                            plot_disc=true
                            );

# Cycle Start!!!!!!!!!!-------------------
function wake_treatment_supress(sim, args...; optargs...)

    # Case: start of simulation -> suppress shedding
    if sim.nt == 1

        # Identify blade elements on which to suppress shedding
        for i in 1:vlm.get_m(rotor)
            HS = vlm.getHorseshoe(rotor, i)
            CP = HS[5]

            if uns.vlm.norm(CP - vlm._get_O(rotor)) <= no_shedding_Rthreshold*R
                push!(omit_shedding, i)
            end
        end
    end

    # Case: sufficient time steps -> enable shedding
    if sim.nt == no_shedding_nstepsthreshold

        # Flag to stop suppressing
        omit_shedding .= -1

    end

    return false
end


# ----------------- 1) VEHICLE DEFINITION --------------------------------------
println("Generating geometry...")


println("Generating vehicle...")

# Generate vehicle
system = vlm.WingSystem()                   # System of all FLOWVLM objects
vlm.addwing(system, "Rotor", rotor)

rotors = [rotor];                           # Defining this rotor as its own system
rotor_systems = (rotors, );                 # All systems of rotors

wake_system = vlm.WingSystem()              # System that will shed a VPM wake
                                            # NOTE: Do NOT include rotor when using the quasi-steady solver
if VehicleType != uns.QVLMVehicle
    vlm.addwing(wake_system, "Rotor", rotor)
end

vehicle = VehicleType(   system;
                            rotor_systems=rotor_systems,
                            wake_system=wake_system
                         );


# ------------- 2) MANEUVER DEFINITION -----------------------------------------
# Non-dimensional translational velocity of vehicle over time
Vvehicle(t) = zeros(3)

# Angle of the vehicle over time
anglevehicle(t) = zeros(3)

# RPM control input over time (RPM over `RPMref`)
RPMcontrol(t) = 1.0

angles = ()                                 # Angle of each tilting system (none)

RPMs = (RPMcontrol, )                       # RPM of each rotor system

maneuver = uns.KinematicManeuver(angles, RPMs, Vvehicle, anglevehicle)


# ------------- 3) SIMULATION DEFINITION ---------------------------------------

Vref = 0.0                                  # Reference velocity to scale maneuver by
RPMref = RPM                                # Reference RPM to scale maneuver by
Vinit = Vref*Vvehicle(0)                    # Initial vehicle velocity
Winit = pi/180*(anglevehicle(1e-6) - anglevehicle(0))/(1e-6*ttot)  # Initial angular velocity

simulation = uns.Simulation(vehicle, maneuver, Vref, RPMref, ttot;
                                                    Vinit=Vinit, Winit=Winit);

# Restart simulation
restart_file = nothing

# NOTE: Uncomment the following line to restart a previous simulation.
#       Point it to a particle field file (with its full path) at a specific
#       time step, and `run_simulation` will start this simulation with the
#       particle field found in the restart simulation.

# restart_file = "/path/to/a/previous/simulation/rotorhover-example_pfield.360"


# ------------- 4) MONITORS DEFINITIONS ----------------------------------------

# Generate rotor monitor
# Generate rotor monitor
monitor_rotor = uns.generate_monitor_rotors(rotors, J, rho, RPM, nsteps;
                                            t_scale=RPM/60,        # Scaling factor for time in plots
                                            t_lbl="Revolutions",   # Label for time axis
                                            save_path=save_path,
                                            save_init_plots=false,
                                            disp_conv=false,
                                            run_name=run_name,
                                            conv_suff=conv_suff_path,
                                            figname="rotor monitor",
                                            )

# Concatenate monitors
monitors = monitor_rotor

# ------------- 5) RUN SIMULATION ----------------------------------------------
println("Running simulation...")

# Concatenate monitors and wake treatment procedure into one runtime function
runtime_function = uns.concatenate(monitors, wake_treatment_supress)

# Run simulation
uns.run_simulation(simulation, nsteps;
                    # ----- SIMULATION OPTIONS -------------
                    Vinf=Vinf,
                    rho=rho, mu=mu, sound_spd=speedofsound,
                    # ----- SOLVERS OPTIONS ----------------
                    p_per_step=p_per_step,
                    max_particles=max_particles,
                    vpm_integration=vpm_integration,
                    vpm_viscous=vpm_viscous,
                    vpm_SFS=vpm_SFS,
                    sigma_vlm_surf=sigma_rotor_surf,
                    sigma_rotor_surf=sigma_rotor_surf,
                    sigma_vpm_overwrite=sigma_vpm_overwrite,
                    sigmafactor_vpmonvlm=sigmafactor_vpmonvlm,
                    vlm_rlx=vlm_rlx,
                    hubtiploss_correction=hubtiploss_correction,
                    shed_starting=shed_starting,
                    shed_unsteady=shed_unsteady,
                    unsteady_shedcrit=unsteady_shedcrit,
                    omit_shedding=omit_shedding,
                    extra_runtime_function=runtime_function,
                    # ----- RESTART OPTIONS -----------------
                    restart_vpmfile=restart_file,
                    # ----- OUTPUT OPTIONS ------------------
                    save_path=save_path,
                    create_savepath = true, 
                    run_name=run_name,
                    save_wopwopin=true,  # <--- Generates input files for PSU-WOPWOP noise analysis
                    save_code=""
                    );

# exit()

import FLOWUnsteady as uns
import FLOWUnsteady: gt, vlm, noise

# Path where to read and save simulation data
sims_path = "/home/sh/WCY/auto_propeller/resource5/3_validation/"

read_path       = joinpath(sims_path, run_name)
save_ww_path    = read_path*"-pww/"

if isdir(save_ww_path)
    # 如果目录已经存在，递归删除它及其内容
    rm(save_ww_path, recursive=true)
end
wopwopbin       = "/home/sh/WCY/T3/vali_Unsteady/validation/utils/WopWop3/wopwop3_linux"


rotorsystems    = [[B]]  

# Aero input parameters
nrevs_st        = 2                    # Number of revolutions to read
nrevs_min       = 6                    # Start reading from this revolution
nsteps_per_rev  = 36                   # Number of steps per revolution in aero solution
num_min         = ceil(Int, nrevs_min*nsteps_per_rev) # Start reading aero files from this step number

if const_solution                       # If constant solution, it overrides to read only the first time step
    nrev_st       = nothing
    nsteps_per_rev = nothing
    num_min     = 1
end

ww_nrevs        = RPM / 60 / 5                     # Number of revolutions in PSU-WOPWOP (18 revs at 5400 RPM gives fbin = 5 Hz)
ww_nsteps_per_rev = max(120, 2*nsteps_per_rev) # Number of steps per revolution in PSU-WOPWOP
const_geometry  = const_solution       # Whether to run PSU-WOPWOP on constant geometry read from num_min
periodic        = true                 # Periodic aerodynamic solution
highpass        = 0.0                  # High pass filter (set this to >0 to get rid of 0th freq in OASPL)

sph_R           = 1.905                # (m) radial distance from rotor hub
sph_nR          = 0                    # Number of microphones in the radial direction
sph_nphi        = 0                    # Number of microphones in the zenith direction
sph_ntht        = 72                   # Number of microphones in the azimuthal direction
sph_thtmin      = 0                    # (deg) first microphone's angle
sph_thtmax      = 360                  # (deg) last microphone's angle
sph_phimax      = 180
sph_rotation    = [90, 0, 0]           # Rotation of grid of microphones

Rmic            = 1.905                # (m) radial distance from rotor hub
anglemic        = 90*pi/180            # (rad) microphone angle from plane of rotation (- below, + above)
                                       # 0deg is at the plane of rotation, 90deg is upstream
microphoneX     = nothing              # Comment and uncomment this to switch from array to single microphone


@time uns.run_noise_wopwop(read_path, run_name, RPM, rho, speedofsound, rotorsystems,
                            ww_nrevs, ww_nsteps_per_rev, save_ww_path, wopwopbin;
                            nrevs=nrevs_st, nsteps_per_rev=nsteps_per_rev,
                            # ---------- OBSERVERS -------------------------
                        sph_R=sph_R,
                            sph_nR=sph_nR, sph_ntht=sph_ntht,
                            sph_nphi=sph_nphi, sph_phimax=sph_phimax,
                            sph_rotation=sph_rotation,
                            sph_thtmin=sph_thtmin, sph_thtmax=sph_thtmax,
                            microphoneX=microphoneX,
                            # ---------- SIMULATION OPTIONS ----------------
                            periodic=periodic,
                            # ---------- INPUT OPTIONS ---------------------
                            num_min=num_min,
                            const_geometry=const_geometry,
                            axisrot="automatic",
                            CW=CW,
                            highpass=highpass,
                            # ---------- OUTPUT OPTIONS --------------------
                            verbose=true, v_lvl=0,
                            prompt=true, debug_paraview=false,
                            debuglvl=0,                     # PSU-WOPWOP debug level (verbose)
                            observerf_name="observergrid",  # .xyz file with observer grid
                            case_name="runcase",            # Name of case to create and run
                            );

read_ww_path   = joinpath(save_ww_path, "runcase")      # Path to PWW's input files
save_vtk_path  = joinpath(read_ww_path, "vtks")         # Where to save VTK files
vtk_str = noise.save_geomwopwop2vtk(read_ww_path, save_vtk_path)

# -------------bpm

# BPM parameters
TE_thickness    = 16.0                 # (deg) trailing edge thickness
noise_correction= 1.00                 # Calibration parameter (1 = no correction)
freq_bins       = uns.BPM.default_f    # Frequency bins (default is one-third octave band)
save_bpm_path   =  read_path*"-bpm/"

if isdir(save_bpm_path)
    # 如果目录已经存在，递归删除它及其内容
    rm(save_bpm_path, recursive=true)
end

rotors = vlm.Rotor[rotor] 

uns.run_noise_bpm(rotors, RPM, Vinf, rho, mu, speedofsound,
                                save_bpm_path;
                                # ---------- OBSERVERS -------------------------
                                sph_R=sph_R,
                                sph_nR=sph_nR, sph_ntht=sph_ntht,
                                sph_nphi=sph_nphi, sph_phimax=sph_phimax,
                                sph_rotation=sph_rotation,
                                sph_thtmin=sph_thtmin, sph_thtmax=sph_thtmax,
                                microphoneX=microphoneX,
                                # ---------- BPM OPTIONS -----------------------
                                noise_correction=noise_correction,
                                TE_thickness=TE_thickness,
                                freq_bins=freq_bins,
                                # ---------- OUTPUT OPTIONS --------------------
                                prompt=true
                                );


# OUTPUT--------------------------
import FLOWUnsteady as uns
import FLOWUnsteady: gt, vlm, noise
using CSV
using DataFrames
import FLOWMath as math

# Path where to read and save simulation data
sims_path = "/home/sh/WCY/auto_propeller/resource5/3_validation/"
dataset_infos = [ # (label, PWW solution, BPM solution, line style, color)
                    ("FLOWUnsteady",
                        joinpath(sims_path, run_name*"-pww/runcase/"),
                        joinpath(sims_path, run_name*"-bpm"),
                        "-", "steelblue"),
                ]

datasets_pww = Dict()     # Stores PWW data in this dictionary
datasets_bpm = Dict()     # Stores BPM data in this dictionary

# Read datasets and stores them in dictionaries
noise.read_data(dataset_infos; datasets_pww=datasets_pww, datasets_bpm=datasets_bpm)

println("Done!")

# Make sure this grid is the same used as an observer by the aeroacoustic solution
sph_R        = 1.905                 # (m) radial distance from rotor hub
sph_nR       = 0
sph_nphi     = 0
sph_ntht     = 72                    # Number of microphones
sph_thtmin   = 0                     # (deg) first microphone's angle
sph_thtmax   = 360                   # (deg) last microphone's angle
sph_phimax   = 180
sph_rotation = [90, 0, 0]            # Rotation of grid of microphones

# Create observer grid
grid = noise.observer_sphere(sph_R, sph_nR, sph_ntht, sph_nphi;
                                thtmin=sph_thtmin, thtmax=sph_thtmax, phimax=sph_phimax,
                                rotation=sph_rotation);


pangle(i) = -180/pi*atan(gt.get_node(grid, i)[1], gt.get_node(grid, i)[2])

using CSV
using DataFrames

# 假设 microphones, pangle, dataset_infos, datasets_pww 已经定义
microphones = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
hash = Dict((round(pangle(mici), digits=1), mici) for mici in 1:sph_ntht)
mics = [hash[deg] for deg in microphones]
P_name = ["TotalAcousticPressure", "LoadingAcousticPressure", "ThicknessAcousticPressure"]

global all_data = DataFrame()

for (i,mici) in enumerate(mics)
    for (di, (lbl, read_path, _, stl, clr)) in enumerate(dataset_infos)
        data = datasets_pww[read_path]["pressure"]
        xi = data["hs"]["ObserverTimes"]
        
        # 第一次循环时初始化Time列
        if i == 1 && di == 1
            global all_data = DataFrame(Time = data["field"][mici, 1, 2:end, xi])
        end
        
        temp_data = DataFrame()
        for Pi in P_name
            yi = data["hs"][Pi]
            P = data["field"][mici, 1, 2:end, yi]
            temp_data[!, Pi * "_" * string(microphones[i])] = P
        end
        
        # 如果不是第一个数据集，需要横向合并（而不是纵向）
        if nrow(all_data) > 0
            global all_data = hcat(all_data, temp_data, makeunique=true)
        else
            global all_data = temp_data
        end
    end
end

# 保存DataFrame为CSV文件

output_file1 = run_name*"/noiseT_$i.csv"
output_file2 = run_name*"/noiseSPL_$i.csv"
CSV.write(output_file1, all_data)

function addSPL(x, y)
    return 10 * log10.(10 .^ (x/10) .+ 10 .^ (y/10))
end
# 
function extract_data_and_save_to_csv(dataset_infos, BPFi_list, BPF, pangle::Function;
  datasets_pww=datasets_pww,
  datasets_bpm=datasets_bpm)
  # 准备一个DataFrame来收集所有数据
  collected_data = DataFrame(Angle = Float64[], Frequency1 = Float64[] , SPL1 = Float64[],SPL_Broad1 =Float64[], Frequency2 = Float64[] , SPL2 = Float64[],SPL_Broad2 =Float64[], OASPL =Float64[])

  fieldname_pww = "spl_spectrum"         # Field to plot
  fieldname_bpm = "spl_spectrum"         # Field to plot

  fieldname_OApww  = "OASPLdB"
  fieldname_OAbpm  = fieldname_OApww

  for (lbl, read_path_pww, read_path_bpm, stl, clr) in dataset_infos

    # Fetch tonal noise
    data_pww = datasets_pww[read_path_pww][fieldname_pww]
    yi = data_pww["hs"]["Total_dB"]
    fi = data_pww["hs"]["Frequency"]

    minf = data_pww["field"][1, 1, 1, fi]                        # Minimum frequency available
    df = data_pww["field"][1, 1, 2, fi] - data_pww["field"][1, 1, 1, fi] # Frequency step
    
    BPFi = BPFi_list[1]
    freqi = ceil(Int, (BPFi*BPF - minf)/df + 1)                           # Frequency index
    freq1 = data_pww["field"][1, 1, freqi, fi] 

    spl_pww1 = data_pww["field"][:, 1, freqi, yi]

    # Fetch broadband noise
    data_bpm = datasets_bpm[read_path_bpm][fieldname_bpm]
    freqs_bpm = datasets_bpm[read_path_bpm]["frequencies"]["field"][:, 1]
    spl_bpm1 = [math.akima(freqs_bpm, data_bpm["field"][:, mici], freq1) for mici in 1:length(spl_pww1)]
    
    spl1 = spl_pww1

    BPFi = BPFi_list[2]
    freqi = ceil(Int, (BPFi*BPF - minf)/df + 1)                           # Frequency index
    freq2 = data_pww["field"][1, 1, freqi, fi] 

    spl_pww2 = data_pww["field"][:, 1, freqi, yi]

    # Fetch broadband noise
    data_bpm = datasets_bpm[read_path_bpm][fieldname_bpm]
    freqs_bpm = datasets_bpm[read_path_bpm]["frequencies"]["field"][:, 1]
    spl_bpm2 = [math.akima(freqs_bpm, data_bpm["field"][:, mici], freq2) for mici in 1:length(spl_pww2)]
    
    spl2 = spl_pww2

    data_pww = datasets_pww[read_path_pww][fieldname_OApww]
    yi = data_pww["hs"]["TotalOASPLdB"]
    oaspl_pww = data_pww["field"][:, 1, 1, yi]
    data_bpm = datasets_bpm[read_path_bpm][fieldname_OAbpm]
    oaspl_bpm = data_bpm["field"][:]
    oaspl = addSPL(oaspl_pww, oaspl_bpm)

    angles = pangle.(1:length(spl1))
    # 将数据添加到DataFrame
    for i in 1:length(spl1)
      push!(collected_data, (Angle = angles[i], Frequency1 = freq1, SPL1 = spl_pww1[i],SPL_Broad1 = spl_bpm1[i], Frequency2 = freq2, SPL2 = spl_pww2[i],SPL_Broad2 = spl_bpm2[i], OASPL = oaspl[i] ))
    end
  end

  # 保存到CSV
  CSV.write(output_file2, collected_data)
end

# 定义参数
BPFi_list = [1,2] # 例如
BPF = 180 # 例如

# 调用函数
extract_data_and_save_to_csv(dataset_infos, BPFi_list, BPF, pangle)

end