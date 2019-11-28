using Clustering;
using Distances;
using LinearAlgebra;
using LossFunctions;
using SpectralClustering;

export magnet_spectral;

function magnet_spectral(model::T, classes::Vector; K::Int = 6, α::Float32 = 0.0f0, clusterIndexUpdateFrequency::Int = 25, dist::PreMetric = SqEuclidean(), innerLoss::SupervisedLoss = L1HingeLoss()) where {T<:MillModel}
	counter = 0; # So that it updates on the first run
	classes = unique(classes);
	clusterCenters = Vector{Vector{Vector{Float32}}}(undef, length(classes));

	function updateClusterIndex(data::A, y::B) where {A<:AbstractArray, B<:AbstractVector}
		function weight(i::Int, neigh, v, m)
			return exp.(-Distances.colwise(dist, m, v));
		end
		for i in 1:length(classes)
			d = data.data[:, y .== classes[i]];
			knnconfig = KNNNeighborhood(d, 10);
			graph = create(knnconfig, weight, d);
			assignments = clusterize(NgLaplacian(3), KMeansClusterizer(K), graph);
			clusterCenters[i] = map(j -> vec(mean(d[:, assignments.assignments .== j], dims = 2)), unique(assignments.assignments))
		end
		counter = clusterIndexUpdateFrequency;
	end

	function μ(r::A)::Vector{Float32} where {A<:AbstractVector}
		centers = vcat(clusterCenters...);
		distances = map(center -> evaluate(dist, r, center), centers)
		return centers[distances .== minimum(distances)][1];
	end

	return function(data::DataSubset)
		outputs = model(getobs(data)).data;
		y = getobs(data).metadata;
		instanceCount = length(y);

		if counter == 0
			updateClusterIndex(outputs, y);
		end

		r(i) = outputs[:, i];
		N = map(i -> evaluate(dist, r(i), μ(r(i))), 1:instanceCount)
		σ² = reduce(+, N) / (instanceCount - 1)
		M = [mapreduce(k -> exp(-evaluate(dist, r(i), classCenters[k]) / (2 * σ²)), +, 1:K) for i in 1:instanceCount, classCenters in clusterCenters]

		numerator(i) = exp((-N[i] / (2 * σ²)) - α);
		denominator(i) = mapreduce(c -> y[i] == c ? 0 : M[i, c], +, 1:length(classes));

		summant(i) = value(innerLoss, log(denominator(i)) - log(numerator(i)));
		sum = mapreduce(i -> summant(i), +, 1:instanceCount);

		counter -= 1;
		return sum / instanceCount;
	end
end
