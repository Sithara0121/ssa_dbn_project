using CSV, DataFrames, Random

# === Step 1: Load the dataset ===
df = CSV.read("cb_labels.csv", DataFrame)

# === Step 2: Clean the data ===
dropmissing!(df)
df.Total_messages = replace(df.Total_messages, 0 => 1)  # avoid division by 0

# === Step 3: Feature Engineering ===
df.Aggression_Ratio = df.Aggressive_Count ./ df.Total_messages
select_features = [:Aggressive_Count, :Total_messages, :Intent_to_Harm, :Peerness, :CB_Label]
dropmissing!(df, select_features)
df.CB_Label = round.(Int, df.CB_Label)  # ensure labels are integers

# === Step 4: Stratified Train-Test Split ===
df_0 = filter(:CB_Label => x -> x == 0, df)
df_1 = filter(:CB_Label => x -> x == 1, df)

function stratified_split(df_class::DataFrame, ratio::Float64)
    n = size(df_class, 1)
    idx = shuffle(1:n)
    cutoff = Int(round(ratio * n))
    return df_class[idx[1:cutoff], :], df_class[idx[cutoff+1:end], :]
end

train_0, test_0 = stratified_split(df_0, 0.8)
train_1, test_1 = stratified_split(df_1, 0.8)

train_df = vcat(train_0, train_1)
test_df = vcat(test_0, test_1)
train_df = train_df[shuffle(1:end), :]
test_df = test_df[shuffle(1:end), :]

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

println("✅ Data loaded, cleaned, normalized, and split successfully.")