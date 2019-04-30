using Flux;
using MLDataPattern;

function CPC(model::T) where {T<:MillModel}
	function diagonalLoss(D::Matrix{Flux.Tracker.TrackedReal})::Vector{Flux.Tracker.TrackedReal}
		return log.(diag(D) .+ eps(Float32)) .- fill(log(sum(D) - sum(diag(D)) + eps(Float32)), size(D, 1));
	end

	return function(data::DataSubset)
		first, second = splitBags(shuffleInBags(getobs(data)));
		D = distanceMatrix(model(first).data, model(second).data, l2squared);
		return mean(diagonalLoss(D));
	end
end
