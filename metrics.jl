# metrics.jl
using Statistics

# function evaluate_predictions(y_pred::Vector{Bool}, y_true::Vector{Bool})
#     tp = sum(y_pred .& y_true)     # True positives
#     tn = sum(.!y_pred .& .!y_true) # True negatives
#     fp = sum(y_pred .& .!y_true)   # False positives
#     fn = sum(.!y_pred .& y_true)   # False negatives
#
#     precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
#     recall    = tp + fn == 0 ? 0.0 : tp / (tp + fn)
#     f1        = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
#     accuracy  = (tp + tn) / (tp + tn + fp + fn)
#
#     println("\n📊 Evaluation Metrics:")
#     println("  Accuracy:  ", round(accuracy * 100, digits=2), "%")
#     println("  Precision: ", round(precision * 100, digits=2), "%")
#     println("  Recall:    ", round(recall * 100, digits=2), "%")
#     println("  F1-score:  ", round(f1 * 100, digits=2), "%")
#     println("  Confusion Matrix:")
#     println("    TP=$tp  FP=$fp")
#     println("    FN=$fn  TN=$tn")
# end
function evaluate_model(dbn, X_test, y_test)
    preds = forward_dbn(dbn, X_test)
    y_true = y_test[1, :]

    # Compute thresholded predictions
    pred_labels = preds .> 0.5

    # Compute precision, recall, F1, etc.
    tp = sum((pred_labels .== 1) .& (y_true .== 1))
    fp = sum((pred_labels .== 1) .& (y_true .== 0))
    fn = sum((pred_labels .== 0) .& (y_true .== 1))
    tn = sum((pred_labels .== 0) .& (y_true .== 0))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    println("📊 Evaluation Metrics:")
    println("Precision: $(round(precision, digits=4))")
    println("Recall:    $(round(recall, digits=4))")
    println("F1 Score:  $(round(f1, digits=4))")
    println("Confusion Matrix:")
    println("TP: $tp | FP: $fp")
    println("FN: $fn | TN: $tn")
end
function precision_recall_f1(preds::AbstractVector{Bool}, labels::AbstractVector{Bool})
    TP = sum(preds .& labels)
    FP = sum(preds .& .!labels)
    FN = sum((.!preds) .& labels)

    precision = TP / (TP + FP + eps())  # Avoid divide by zero
    recall = TP / (TP + FN + eps())
    f1 = 2 * precision * recall / (precision + recall + eps())

    return precision, recall, f1
end


