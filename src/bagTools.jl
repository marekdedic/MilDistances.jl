using Mill;
using Random;

function shuffleInBags(node::BagNode)::BagNode
	bags = ScatteredBags(shuffle.(node.bags.bags));
	return BagNode(node.data, bags, node.metadata);
end

function splitBags(node::BagNode)::Tuple{BagNode, BagNode}
	first = Vector{Vector{Int}}();
	second = Vector{Vector{Int}}();
	for bag in node.bags
		if length(bag) < 2
			continue;
		end
		delim = rand(1:length(bag) - 1);
		push!(first, bag[1:delim]);
		push!(second, bag[delim + 1:end]);
	end
	function toRanges(vec::Vector{Vector{Int}})::Vector{UnitRange{Int}}
		ret = Vector{UnitRange{Int}}();
		counter = 1
		for x in vec
			push!(ret, counter:counter + length(x) - 1);
			counter += length(x);
		end
		return ret;
	end
	return (BagNode(node.data[vcat(first...)], toRanges(first), node.metadata), BagNode(node.data[vcat(second...)], toRanges(second), node.metadata))
end
