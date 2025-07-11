using CSV, DataFrames, Random

# === Step 1: Load the dataset ===
df = CSV.read("cb_labels.csv", DataFrame)
println("📥 Loaded data: $(size(df, 1)) rows, $(size(df, 2)) columns")
first(df, 5) |> println

# === Step 2: Clean the data ===
dropmissing!(df)
df.Total_messages = replace(df.Total_messages, 0 => 1)  # avoid division by 0
println("\n🧹 After cleaning: $(size(df, 1)) rows remain")
first(df, 5) |> println

# === Step 3: Feature Engineering ===
df.Aggression_Ratio = df.Aggressive_Count ./ df.Total_messages
select_features = [:Aggressive_Count, :Total_messages, :Intent_to_Harm, :Peerness, :CB_Label]
dropmissing!(df, select_features)
df.CB_Label = round.(Int, df.CB_Label)
println("\n🛠️ After feature engineering:")
first(df[:, vcat(select_features, [:Aggression_Ratio])], 5) |> println

# === Step 4: Stratified Train-Test Split ===
df_0 = filter(:CB_Label => x -> x == 0, df)
df_1 = filter(:CB_Label => x -> x == 1, df)
println("\n📊 Class distribution before split: Class 0 = $(size(df_0, 1)), Class 1 = $(size(df_1, 1))")

function stratified_split(df_class::DataFrame, ratio::Float64)
    n = size(df_class, 1)
    idx = shuffle(1:n)
    cutoff = Int(round(ratio * n))
    return df_class[idx[1:cutoff], :], df_class[idx[cutoff+1:end], :]
end

train_0, test_0 = stratified_split(df_0, 0.8)
train_1, test_1 = stratified_split(df_1, 0.8)

# === Step 4.1: Oversample minority class in training set ===
function oversample_minority(df_major::DataFrame, df_minor::DataFrame)
    n_major = size(df_major, 1)
    n_minor = size(df_minor, 1)
    needed = n_major - n_minor
    oversampled_rows = df_minor[rand(1:n_minor, needed), :]
    return vcat(df_major, df_minor, oversampled_rows)
end

if size(train_0, 1) > size(train_1, 1)
    train_df = oversample_minority(train_0, train_1)
elseif size(train_1, 1) > size(train_0, 1)
    train_df = oversample_minority(train_1, train_0)
else
    train_df = vcat(train_0, train_1)
end

test_df = vcat(test_0, test_1)
train_df = train_df[shuffle(1:end), :]
test_df = test_df[shuffle(1:end), :]

train_counts = combine(groupby(train_df, :CB_Label), nrow => :Count)
test_counts = combine(groupby(test_df, :CB_Label), nrow => :Count)
println("\n🧪 After split and balancing:")
println("Train class distribution:\n$train_counts")
println("Test class distribution:\n$test_counts")
println("First 5 rows of training data:")
first(train_df, 5) |> println

# === Step 5: Normalize Features (Min-Max Scaling) ===
function minmax_scale(data::Matrix{Float64})
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)
    scaled = (data .- min_vals) ./ (max_vals .- min_vals .+ eps())
    return scaled
end

X_train_raw = Matrix(train_df[:, [:Aggression_Ratio, :Intent_to_Harm, :Peerness]])
X_test_raw = Matrix(test_df[:, [:Aggression_Ratio, :Intent_to_Harm, :Peerness]])

X_train = Array{Float32}(minmax_scale(X_train_raw))'
X_test = Array{Float32}(minmax_scale(X_test_raw))'

y_train = Array{Float32}(Matrix(train_df[:, [:CB_Label]])')
y_test = Array{Float32}(Matrix(test_df[:, [:CB_Label]])')

println("\n📐 Normalization complete")
println("X_train shape: ", size(X_train))
println("y_train shape: ", size(y_train))
