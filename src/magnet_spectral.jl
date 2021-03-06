using Clustering;
using Distances;
using Flux;
using LinearAlgebra;
using LossFunctions;
using SpectralClustering;

export magnet_spectral;

function magnet_spectral(model::T, classes::Vector; K::Int = 2, α::Float32 = 0.0f0, clusterIndexUpdateFrequency::Int = 25, dist::PreMetric = SqEuclidean(), innerLoss::SupervisedLoss = L1HingeLoss()) where {T<:AbstractMillModel}
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
		if isempty(centers)
			@warn "clusterCenters are empty"
			return fill(0.0f0, length(r));
		end
		distances = map(center -> evaluate(dist, r, center), centers)
		return centers[argmin(distances)];
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
		σ² = instanceCount == 1 ? N[1] : reduce(+, N) / (instanceCount - 1);
		M = [mapreduce(k -> exp(-evaluate(dist, r(i), classCenters[k]) / (2 * σ²)), +, 1:K) for i in 1:instanceCount, classCenters in clusterCenters]

		numerator(i) = exp((-N[i] / (2 * σ²)) - α);
		denominator(i) = mapreduce(c -> y[i] == c ? 0 : M[i, c], +, 1:length(classes));

		logNum(i) = iszero(numerator(i)) ? begin @warn "numerator = 0"; param(0.0f0) end : log(numerator(i));
		logDen(i) = iszero(denominator(i)) ? begin @warn "denominator = 0"; param(0.0f0) end : log(denominator(i));

		summant(i) = value(innerLoss, logDen(i) - logNum(i));
		sum = mapreduce(i -> summant(i), +, 1:instanceCount);

		counter -= 1;
		ret = sum / instanceCount;
		if isinf(ret)
			if iszero(instanceCount)
				@warn "instanceCount = 0";
			end
			return param(0.0f0);
		end
		if isnan(ret)
			@warn "loss is NaN";
			return param(0.0f0);
		end
		return ret;
	end
end
