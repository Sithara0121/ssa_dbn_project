# === Include Modules ===
include("data_preparation.jl")
include("ssa_dbn_training.jl")
include("metrics.jl")
include("ssa_optimizer.jl")
include("ssa_log_plot.jl")
include("cross_validation.jl")
using BSON  # for saving/loading models

# === Step 1: Train with Fixed Configuration ===
println("🔧 Running fixed DBN training before SSA...")
dbn = run_dbn_training(X_train, y_train, X_test, y_test; h1=6, h2=3, lr=0.01f0, epochs=10)

# === Step 2: Evaluate Fixed DBN ===
println("📊 Evaluation Metrics for Fixed DBN:")
evaluate_model(dbn, X_test, y_test)

# === Step 3: Optional K-Fold Cross-Validation ===
println("🧪 Running K-Fold Cross-Validation (5 folds)...")
cross_validate_dbn(X_train, y_train; k=5, h1=6, h2=3, lr=0.01f0, epochs=10)

# === Step 4: Run SSA Optimization ===
println("🚀 Starting SSA optimization...")
best_dbn = ssa_optimize(X_train, y_train, X_test, y_test; num_agents=5, max_iter=5)

if best_dbn === nothing
    println("❌ SSA optimization did not return a model.")
    exit()
end

# === Step 5: Plot SSA Optimization Accuracy Log ===
println("📈 Plotting SSA optimization accuracy...")
plot_ssa_accuracy_log()

# === Step 6: Save Models ===
println("💾 Saving fixed DBN model to 'best_fixed_model.bson'...")
BSON.@save "best_fixed_model.bson" dbn

println("💾 Saving SSA-optimized DBN model to 'best_ssa_model.bson'...")
BSON.@save "best_ssa_model.bson" best_dbn

# === Step 7: Re-evaluate SSA-Best Model ===
println("📊 Re-evaluating SSA-optimized model:")
evaluate_model(best_dbn, X_test, y_test)
