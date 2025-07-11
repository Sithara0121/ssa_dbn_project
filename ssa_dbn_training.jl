# DBN Training and Evaluation in Julia (SSA-Ready with Logs)

using Flux, Statistics, Random, LinearAlgebra, Distributions

# === Restricted Boltzmann Machine ===
struct RBM
    W::Array{Float32, 2}
    vbias::Array{Float32, 1}
    hbias::Array{Float32, 1}
end

sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))

function sample_bernoulli(p::AbstractArray{Float32})
    rand.(Bernoulli.(p))
end

function train_rbm(rbm::RBM, data::AbstractMatrix{Float32}, lr::Float32, epochs::Int)
    data = Matrix(data)  # ensure it's not an Adjoint
    println("⏳ Starting RBM training with $(size(rbm.W, 2)) hidden units for $epochs epochs...")
    for epoch in 1:epochs
        for x in eachcol(data)
            h_prob = sigmoid(rbm.W' * x .+ rbm.hbias)
            h_state = sample_bernoulli(h_prob)
            v_recon = sigmoid(rbm.W * h_state .+ rbm.vbias)
            h_recon = sigmoid(rbm.W' * v_recon .+ rbm.hbias)

            rbm.W .+= lr .* ((x * h_prob') .- (v_recon * h_recon'))
            rbm.vbias .+= lr .* (x .- v_recon)
            rbm.hbias .+= lr .* (h_prob .- h_recon)
        end
        println("  🔁 Finished epoch $epoch")
    end
    println("✅ RBM training complete.")
    return rbm
end

# === Deep Belief Network ===
mutable struct DBN
    rbms::Vector{RBM}
    classifier::Chain
end

function forward_dbn(dbn::DBN, x::AbstractArray)
    for rbm in dbn.rbms
        x = sigmoid(rbm.W' * x .+ rbm.hbias)
    end
    return dbn.classifier(x)
end

function train_dbn(dbn::DBN, x_train, y_train, lr::Float32, epochs::Int)
    println("⏳ Starting supervised training of DBN classifier...")

    model = dbn.classifier
    opt = Flux.ADAM(lr)
    state = Flux.setup(opt, model)

    for epoch in 1:epochs
        for i in 1:size(x_train, 2)
            x = x_train[:, i]
            y = y_train[:, i]

            # Compute gradients with respect to model only
            loss, back = Flux.withgradient(model) do m
                ŷ = forward_dbn(DBN(dbn.rbms, m), x)
                Flux.binarycrossentropy(ŷ, y)
            end

            Flux.update!(state, model, back)
        end
        println("  🔁 Supervised training epoch $epoch completed")
    end

    println("✅ Classifier fine-tuning complete.")
end

# === Training Example ===
function run_dbn_training(X_train, y_train, X_test, y_test; h1=8, h2=4, lr=0.01f0, epochs=5,return_accuracy=false)
   # println("📐 Input size: ", size(X_train, 1), " features")
    #println("📊 Training set: ", size(X_train, 2), " samples")

    input_dim = size(X_train, 1)
    println("🧱 Initializing DBN with hidden sizes: $h1 → $h2")
    rbm1 = RBM(randn(Float32, input_dim, h1), zeros(Float32, input_dim), zeros(Float32, h1))
    rbm2 = RBM(randn(Float32, h1, h2), zeros(Float32, h1), zeros(Float32, h2))

    # Greedy pretraining
    println("⚙️ Starting unsupervised pretraining of RBMs...")
    train_rbm(rbm1, X_train, lr, epochs)
    h1_out = sigmoid.(rbm1.W' * X_train .+ rbm1.hbias)
    train_rbm(rbm2, h1_out, lr, epochs)

    # Classifier
    println("🧠 Building classifier and fine-tuning the network...")
    classifier = Chain(Dense(h2, 1), sigmoid)
    dbn = DBN([rbm1, rbm2], classifier)
    train_dbn(dbn, X_train, y_train, lr, epochs)

    # Evaluation
    println("🔍 Evaluating on test set...")
    preds = forward_dbn(dbn, X_test) .> 0.5
    acc = mean((preds .== (y_test .== 1)))
    if return_accuracy
        return acc
    end
    println("✅ Test Accuracy: ", round(acc * 100, digits=2), "%")
    #println("✅ Test Accuracy: ", round(acc * 100, digits=2), "%")
end

# === Example Run (replace with real data) ===
# run_dbn_training(X_train, y_train, X_test, y_test; h1=6, h2=3, lr=0.01f0, epochs=10)
