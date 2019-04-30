function hinge(x::T) where T<:Number
	return max(zero(T), one(T) - x);
end
