using Flux, Statistics, Random, LinearAlgebra, Distributions
include("metrics.jl")

# === RBM and DBN Structures ===
struct RBM
    W::Array{Float32, 2}
    vbias::Array{Float32, 1}
    hbias::Array{Float32, 1}
end

mutable struct DBN
    rbms::Vector{RBM}
    classifier::Chain
end

# === Utilities ===
sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp.(-x))

function sample_bernoulli(p::AbstractArray{Float32})
    rand.(Bernoulli.(p))
end

# === RBM Training ===
function train_rbm(rbm::RBM, data::AbstractMatrix{Float32}, lr::Float32, epochs::Int)
    data = Matrix(data)
    println("⏳ Training RBM with $(size(rbm.W, 2)) hidden units for $epochs epochs...")
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
        println("  🔁 RBM Epoch $epoch complete")
    end
    println("✅ RBM training complete.")
    return rbm
end

# === DBN Forward ===
function forward_dbn(dbn::DBN, x::AbstractArray)
    for rbm in dbn.rbms
        x = sigmoid(rbm.W' * x .+ rbm.hbias)
    end
    return dbn.classifier(x)
end

# === DBN Supervised Training ===
function train_dbn(dbn::DBN, x_train, y_train, lr::Float32, epochs::Int)
    println("⏳ Starting supervised fine-tuning...")

    model = dbn.classifier
    opt = Flux.ADAM(lr)
    state = Flux.setup(opt, model)

    for epoch in 1:epochs
        for i in 1:size(x_train, 2)
            x = x_train[:, i]
            y = y_train[:, i]

            loss, back = Flux.withgradient(model) do m
                ŷ = forward_dbn(DBN(dbn.rbms, m), x)
                Flux.binarycrossentropy(ŷ, y)
            end

            Flux.update!(state, model, back[1])
        end
        println("  🔁 Supervised epoch $epoch done")
    end
    println("✅ Fine-tuning complete.")
end

# === Main Training Driver ===
function run_dbn_training(X_train, y_train, X_test, y_test;
                          h1=8, h2=4, lr=0.01f0, epochs=5, return_accuracy=false)

    input_dim = size(X_train, 1)
    n_train = size(X_train, 2)
    n_test = size(X_test, 2)

    println("📐 Input feature count: $input_dim")
    println("📊 Training samples: $n_train, Test samples: $n_test")
    println("🔎 First 3 training labels: ", y_train[:, 1:3])
    println("🔎 First 3 training features (per feature row):")
    for i in 1:input_dim
        println("  Feature $i: ", X_train[i, 1:3])
    end

    println("\n🧱 Initializing DBN with hidden sizes: $h1 → $h2")
    rbm1 = RBM(randn(Float32, input_dim, h1), zeros(Float32, input_dim), zeros(Float32, h1))
    rbm2 = RBM(randn(Float32, h1, h2), zeros(Float32, h1), zeros(Float32, h2))

    println("⚙️ Starting unsupervised pretraining of RBM 1...")
    train_rbm(rbm1, X_train, lr, epochs)
    h1_out = sigmoid.(rbm1.W' * X_train .+ rbm1.hbias)

    println("🧠 RBM 1 output shape: ", size(h1_out))
    println("🔎 Sample output from RBM 1: ", h1_out[:, 1:3])

    println("⚙️ Starting unsupervised pretraining of RBM 2...")
    train_rbm(rbm2, h1_out, lr, epochs)

    println("🧠 Building classifier and fine-tuning the DBN...")
    classifier = Chain(Dense(h2, 1), sigmoid)
    dbn = DBN([rbm1, rbm2], classifier)

    train_dbn(dbn, X_train, y_train, lr, epochs)

    println("🔍 Evaluating on test set...")
    preds = forward_dbn(dbn, X_test) .> 0.5
    acc = mean((preds .== (y_test .== 1)))
    println("✅ Test Accuracy: ", round(acc * 100, digits=2), "%")

    n_show = min(5, size(preds, 1))
    println("🔎 First $n_show predictions vs labels:")
    println("  Predictions: ", preds[1:n_show])
    println("  Actual:      ", y_test[1, 1:n_show])


    if return_accuracy
        return acc
    else
        return dbn
    end
end
