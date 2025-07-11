# Retrain Final DBN Model with Best SSA Parameters

include("data_preparation.jl")
include("ssa_dbn_training.jl")

# Best parameters found by SSA
best_h1 = 6
best_h2 = 5
best_lr = 0.0106f0
best_epochs = 10  # You can increase this if desired

println("\n🎯 Retraining final model with best parameters:")
println("  h1 = $best_h1, h2 = $best_h2, lr = $best_lr, epochs = $best_epochs")

final_acc = run_dbn_training(X_train, y_train, X_test, y_test;
                             h1=best_h1,
                             h2=best_h2,
                             lr=best_lr,
                             epochs=best_epochs,
                             return_accuracy=true)

println("\n✅ Final Retrained Accuracy: $(round(final_acc * 100, digits=2))%")
