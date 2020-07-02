using Distances;
using Flux;
using MLDataPattern;
using Statistics;

export CPC;

function CPC(model::T, dist::PreMetric = SqEuclidean()) where {T<:AbstractMillModel}
	function diagonalLoss(D::T) where T<:AbstractMatrix
		return log.(diag(D) .+ eps(Float32)) .- fill(log(sum(D) - sum(diag(D)) + eps(Float32)), size(D, 1));
	end

	return function(data::DataSubset)
		first, second = splitBags(shuffleInBags(getobs(data)));
		# https://github.com/FluxML/Tracker.jl/issues/59
		# Workaround:
		X = model(first).data;
		Y = model(second).data;
		D = [evaluate(dist, X[:, i], Y[:, j]) for i in 1:size(X, 2), j in 1:size(Y, 2)]
		# D = pairwise(dist, model(first).data, model(second).data, dims = 2);
		return mean(diagonalLoss(D));
	end
end
