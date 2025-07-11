using Plots, CSV, DataFrames

function plot_ssa_accuracy_log(csv_path="ssa_log.csv", output_path=nothing)
    df = CSV.read(csv_path, DataFrame)
    accuracy = df.accuracy
    plt = plot(accuracy,
         title="SSA Optimization Accuracy",
         xlabel="Iteration",
         ylabel="Accuracy",
         legend=false,
         lw=2,
         marker=:circle)
    if output_path !== nothing
        savefig(plt, output_path)
    end
    return plt
end
