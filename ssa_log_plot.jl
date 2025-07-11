# Plot SSA Optimization Accuracy Progress

using CSV, DataFrames, Plots

# Load SSA log
log_df = CSV.read("ssa_log.csv", DataFrame)

# Add iteration column (based on num_agents)
num_agents = 5  # Change this to match your SSA run
log_df.iteration = repeat(1:(nrow(log_df) ÷ num_agents), inner=num_agents)

# Plot accuracy vs iteration
plot(log_df.iteration, log_df.accuracy,
     seriestype = :scatter,
     markersize = 4,
     color = :blue,
     xlabel = "Iteration",
     ylabel = "Accuracy",
     title = "SSA Optimization Progress",
     legend = false)

savefig("ssa_accuracy_plot.png")
println("✅ Plot saved as 'ssa_accuracy_plot.png'")
