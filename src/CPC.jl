using Distances;
using Flux;
using MLDataPattern;

function CPC(model::T, dist::PreMetric = SqEuclidean()) where {T<:MillModel}
	function diagonalLoss(D::Matrix{Flux.Tracker.TrackedReal})::Vector{Flux.Tracker.TrackedReal}
		return log.(diag(D) .+ eps(Float32)) .- fill(log(sum(D) - sum(diag(D)) + eps(Float32)), size(D, 1));
	end

	return function(data::DataSubset)
		first, second = splitBags(shuffleInBags(getobs(data)));
		D = pairwise(dist, model(first).data, model(second).data, dims = 2);
		return mean(diagonalLoss(D));
	end
end
