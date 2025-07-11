using Random, Statistics
include("ssa_dbn_training.jl")
include("metrics.jl")

function cross_validate_dbn(X, y; k=5, h1=6, h2=3, lr=0.01f0, epochs=10)
    n = size(X, 2)
    indices = shuffle(collect(1:n))
    fold_size = div(n, k)
    accs = Float64[]
    precisions = Float64[]
    recalls = Float64[]
    f1s = Float64[]

    for i in 1:k
        test_idx = indices[(i-1)*fold_size + 1 : i == k ? end : i*fold_size]
        train_idx = setdiff(indices, test_idx)

        X_train = X[:, train_idx]
        y_train = y[:, train_idx]
        X_test = X[:, test_idx]
        y_test = y[:, test_idx]

        # Train and get the model
        dbn = run_dbn_training(X_train, y_train, X_test, y_test;
                               h1=h1, h2=h2, lr=lr, epochs=epochs, return_accuracy=false)

        # Predict
        preds = forward_dbn(dbn, X_test)
        preds_bin = preds .> 0.5
        acc = mean((preds_bin .== (y_test .== 1)))
        push!(accs, acc)

        # Metrics
        p, r, f = precision_recall_f1(vec(preds_bin), vec(y_test .== 1))
        push!(precisions, p)
        push!(recalls, r)
        push!(f1s, f)

        println("Fold $i: Accuracy=$(round(acc*100, digits=2))%, Precision=$(round(p, digits=2)), Recall=$(round(r, digits=2)), F1=$(round(f, digits=2))")
    end

    println("\n📊 Cross-Validation Results (average over $k folds):")
    println("  Accuracy:  $(round(mean(accs)*100, digits=2))%")
    println("  Precision: $(round(mean(precisions), digits=2))")
    println("  Recall:    $(round(mean(recalls), digits=2))")
    println("  F1-score:  $(round(mean(f1s), digits=2))")
end
