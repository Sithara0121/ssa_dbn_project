using BSON
include("data_preparation.jl")
include("ssa_dbn_training.jl")

# Best SSA params
h1 = 6
h2 = 5
lr = 0.0106f0
epochs = 10

println("🎯 Training model to save...")
rbm1 = RBM(randn(Float32, size(X_train, 1), h1), zeros(Float32, size(X_train, 1)), zeros(Float32, h1))
rbm2 = RBM(randn(Float32, h1, h2), zeros(Float32, h1), zeros(Float32, h2))
classifier = Chain(Dense(h2, 1), sigmoid)
dbn = DBN([rbm1, rbm2], classifier)

train_rbm(rbm1, X_train, lr, epochs)
h1_out = sigmoid.(rbm1.W' * X_train .+ rbm1.hbias)
train_rbm(rbm2, h1_out, lr, epochs)
train_dbn(dbn, X_train, y_train, lr, epochs)

# Save
BSON.@save "final_dbn.bson" dbn
println("✅ Model saved to 'final_dbn.bson'")

# Load and test
BSON.@load "final_dbn.bson" dbn
preds = forward_dbn(dbn, X_test) .> 0.5
acc = mean((preds .== (y_test .== 1)))
println("📦 Reloaded model accuracy: ", round(acc * 100, digits=2), "%")
