export triplet;

function triplet(model::T; c::Float32 = 1.0f0, innerLoss::SupervisedLoss = L1HingeLoss(), dist::PreMetric = SqEuclidean()) where {T<:MillModel}
	return function(data::DataSubset)
		y = getobs(data).metadata;
		yMat = y .== y';
		η = yMat; # TODO: Add target neighbour selection
		D = pairwise(dist, model(getobs(data)).data, dims = 2);
		clusterTerm = sum(
			i->sum(
				j->η[i, j] * D[i, j],
				1:length(y)),
			1:length(y))
		nonClusterTerm = sum(
			i->sum(
				j->sum(
					l-> η[i, j] * (1 - yMat[i, l]) * value(innerLoss, D[i, l] - D[i, j]),
					1:length(y)),
				1:length(y)),
			1:length(y));

		return clusterTerm + c * nonClusterTerm;
	end
end
