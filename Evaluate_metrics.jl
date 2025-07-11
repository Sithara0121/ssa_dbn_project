# Evaluate DBN with Confusion Matrix, Precision, Recall, F1

using Statistics
include("data_preparation.jl")
include("ssa_dbn_training.jl")

# Use best-known parameters (adjust if needed)
h1 = 6
h2 = 5
lr = 0.0106f0
epochs = 10

# Retrain final model
dbn_acc = run_dbn_training(X_train, y_train, X_test, y_test;
                           h1=h1, h2=h2, lr=lr, epochs=epochs)

# Predict all test samples for detailed metrics
dbn = let
    rbm1 = RBM(randn(Float32, size(X_train, 1), h1), zeros(Float32, size(X_train, 1)), zeros(Float32, h1))
    rbm2 = RBM(randn(Float32, h1, h2), zeros(Float32, h1), zeros(Float32, h2))
    classifier = Chain(Dense(h2, 1), sigmoid)
    dbn = DBN([rbm1, rbm2], classifier)
    train_rbm(rbm1, X_train, lr, epochs)
    h1_out = sigmoid.(rbm1.W' * X_train .+ rbm1.hbias)
    train_rbm(rbm2, h1_out, lr, epochs)
    train_dbn(dbn, X_train, y_train, lr, epochs)
    dbn
end

# Predictions and ground truth
preds = forward_dbn(dbn, X_test) .> 0.5
truth = y_test .== 1

TP = sum(preds .& truth)
TN = sum(.!preds .& .!truth)
FP = sum(preds .& .!truth)
FN = sum(.!preds .& truth)

precision = TP / (TP + FP + eps())
recall = TP / (TP + FN + eps())
f1 = 2 * precision * recall / (precision + recall + eps())

println("\n📊 Evaluation Metrics:")
println("  Confusion Matrix:")
println("    TP = $TP, FP = $FP")
println("    FN = $FN, TN = $TN")
println("  Precision = ", round(precision, digits=4))
println("  Recall    = ", round(recall, digits=4))
println("  F1 Score  = ", round(f1, digits=4))
