using Plots, CSV, DataFrames

function plot_ssa_accuracy_log(csv_path="ssa_log.csv")
    df = CSV.read(csv_path, DataFrame)
    accuracy = df.accuracy
    plot(accuracy,
         title="SSA Optimization Accuracy",
         xlabel="Iteration",
         ylabel="Accuracy",
         legend=false,
         lw=2,
         marker=:circle)
end
