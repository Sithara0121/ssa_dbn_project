include("data_preparation.jl")
include("ssa_dbn_training.jl")

# Run DBN Training
run_dbn_training(X_train, y_train, X_test, y_test; h1=6, h2=3, lr=0.01f0, epochs=10)
include("ssa_optimizer.jl")

# Start SSA tuning
ssa_optimize(X_train, y_train, X_test, y_test; num_agents=5, max_iter=5)
