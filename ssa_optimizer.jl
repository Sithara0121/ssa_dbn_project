# Salp Swarm Algorithm for Optimizing DBN Hyperparameters with Logging

using Random, Statistics, CSV, DataFrames
include("ssa_dbn_training.jl")

function objective(params::Vector{Float64}, X_train, y_train, X_test, y_test)
    h1 = round(Int, clamp(params[1], 3, 20))
    h2_raw = clamp(params[2], 2.0, 15.0)
    h2 = round(Int, min(h2_raw, h1 - 1))  # force h2 < h1
    h2 = max(h2, 2)  # ensure h2 ≥ 2
    lr = clamp(params[3], 0.0005, 0.1)
    epochs = 5

    try
        println("🚀 Trying h1=$h1, h2=$h2, lr=$(round(lr, digits=4))")
        acc = run_dbn_training(X_train, y_train, X_test, y_test;
                               h1=h1, h2=h2, lr=Float32(lr), epochs=epochs, return_accuracy=true)
        println("✅ Accuracy: $(round(acc * 100, digits=2))%\n")
        return 1.0 - acc
    catch e
        println("⚠️ Error for params: $params → $e")
        return 1.0
    end
end

function ssa_optimize(X_train, y_train, X_test, y_test; num_agents=10, max_iter=10)
    dim = 3
    lb = [3.0, 2.0, 0.0005]
    ub = [20.0, 15.0, 0.1]

    population = [lb .+ rand(dim) .* (ub .- lb) for _ in 1:num_agents]
    fitness = [objective(p, X_train, y_train, X_test, y_test) for p in population]

    best_idx = argmin(fitness)
    best_pos = copy(population[best_idx])
    best_fit = fitness[best_idx]

    log_rows = DataFrame(h1=Int[], h2=Int[], lr=Float64[], accuracy=Float64[])

    for iter in 1:max_iter
        #println("\n🌊 SSA Iteration $iter")
        c1 = 2 * exp(-((4 * iter / max_iter)^2))

        for i in 1:num_agents
            if i == 1
                population[i] = best_pos .+ c1 .* ((ub .- lb) .* rand(dim) .+ lb)
            else
                population[i] = 0.5 .* (population[i] .+ population[i - 1])
            end

            population[i] = clamp.(population[i], lb, ub)
            f = objective(population[i], X_train, y_train, X_test, y_test)
            fitness[i] = f

            acc = 1.0 - f
            push!(log_rows, (round(Int, population[i][1]), round(Int, min(population[i][2], population[i][1] - 1)), population[i][3], acc))
        end

        best_idx = argmin(fitness)
        if fitness[best_idx] < best_fit
            best_fit = fitness[best_idx]
            best_pos = copy(population[best_idx])
        end

        println("🔥 Best accuracy so far: $(round((1.0 - best_fit) * 100, digits=2))%")
    end

    println("\n🎯 SSA Optimization Complete")
    println("Best Hyperparameters: h1=$(round(Int, best_pos[1])), h2=$(round(Int, min(best_pos[2], best_pos[1] - 1))), lr=$(round(best_pos[3], digits=4))")
    println("Best Accuracy: $(round((1.0 - best_fit) * 100, digits=2))%")

    CSV.write("ssa_log.csv", log_rows)
    println("📄 Optimization log saved to 'ssa_log.csv'")
end
