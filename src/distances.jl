function distanceMatrix(X::A, Y::B, dist::Function) where {A<:AbstractMatrix, B<:AbstractMatrix}
	return [dist(X[:, i], Y[:, j]) for i in 1:size(X, 2), j in 1:size(Y, 2)];
end

function distanceMatrix(X::A, dist::Function) where {A<:AbstractMatrix}
	return distanceMatrix(X, X, dist);
end

function l2(x::A, y::B) where {A<:AbstractVector, B<:AbstractVector}
	return norm(x - y);
end

function l2squared(x::A, y::B) where {A<:AbstractVector, B<:AbstractVector}
	return norm(x - y)^2;
end
