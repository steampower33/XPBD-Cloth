#include "context.h"
#include "cloth_sim_data.h"

#include "cloth_gpu_profiler.h"

ClothGpuProfiler::ClothGpuProfiler(Context& context, ClothData& simData)
	: context_(context)
{
	CreateQueryPool();

	slots_per_compute_ =
		1
		+ simData.substeps *
		(slots_pre_hashing + slots_spatial_hashing_
			+ simData.iterations * slots_per_iteration_
			+ slots_post_iteration_)
		+ slots_calculate_normals_
		+ 1;
}

ClothGpuProfiler::~ClothGpuProfiler()
{

}

void ClothGpuProfiler::CreateQueryPool() {
	vk::QueryPoolCreateInfo queryInfo = {};
	queryInfo.queryType = vk::QueryType::eTimestamp;
	queryInfo.queryCount = 2048;

	timestamp_pool_ = context_.device_.createQueryPool(queryInfo);
}

void ClothGpuProfiler::CalculateGpuTime(ClothData& simData)
{
	float nsPerTick = context_.physical_device_.getProperties().limits.timestampPeriod;
	float toMs = nsPerTick / 1e6f;

	if (timestamp_steps_ <= 0) return;

	uint32_t numTimestamp = timestamp_steps_;
	std::vector<uint64_t> ts(numTimestamp);

	VkResult res = vkGetQueryPoolResults(
		static_cast<VkDevice>(*context_.device_),
		static_cast<VkQueryPool>(*timestamp_pool_),
		0, numTimestamp,
		ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);

	auto delta_ms = [&](uint32_t i0, uint32_t i1) {
		return (ts[i1] - ts[i0]) * toMs;
		};

	float tWind = 0.0f;
	float tIntegrate = 0.0f;
	float tClearLambdas = 0.0f;
	float tHashBuild = 0.0f;
	float tRadixSort = 0.0f;
	float tBuildCell = 0.0f;
	float tBuildNeighbor = 0.0f;
	float tSolveStretch = 0.0f;
	float tSolveShear = 0.0f;
	float tSolveBend = 0.0f;
	float tSolveArea = 0.0f;
	float tSolveSelfCollision = 0.0f;
	float tApplyDeltas = 0.0f;
	float tSolveLRA = 0.0f;
	float tCollideSdf = 0.0f;
	float tUpdate = 0.0f;
	float tCalculateNormals = 0.0f;

	// Walk the timestamps in the exact order ClothSimPass::RecordCompute writes them.
	// The broad phase only writes its slots on substeps where (sub % broadphase_interval_ == 0),
	// so the per-substep slot count varies and the cursor must advance per substep.
	uint32_t expected = 1;
	for (uint32_t sub = 0; sub < simData.substeps; sub++)
	{
		expected += slots_pre_hashing;
		if (sub % simData.broadphase_interval_ == 0)
			expected += slots_spatial_hashing_;
		expected += simData.iterations * slots_per_iteration_ + slots_post_iteration_;
	}
	expected += slots_calculate_normals_ + 1;

	// Settings changed between recording and readback: the layout no longer matches.
	if (expected != numTimestamp) return;

	uint32_t cur = 1;
	for (uint32_t sub = 0; sub < simData.substeps; sub++)
	{
		tWind += delta_ms(cur + 0, cur + 1);
		tIntegrate += delta_ms(cur + 2, cur + 3);
		tClearLambdas += delta_ms(cur + 4, cur + 5);
		cur += slots_pre_hashing;

		if (sub % simData.broadphase_interval_ == 0)
		{
			tHashBuild += delta_ms(cur + 0, cur + 1);
			tRadixSort += delta_ms(cur + 2, cur + 3);
			tBuildCell += delta_ms(cur + 4, cur + 5);
			tBuildNeighbor += delta_ms(cur + 6, cur + 7);
			cur += slots_spatial_hashing_;
		}

		for (uint32_t it = 0; it < simData.iterations; it++)
		{
			tSolveStretch += delta_ms(cur + 0, cur + 1);
			tSolveShear += delta_ms(cur + 2, cur + 3);
			tSolveBend += delta_ms(cur + 4, cur + 5);
			tSolveArea += delta_ms(cur + 6, cur + 7);
			tSolveSelfCollision += delta_ms(cur + 8, cur + 9);
			tApplyDeltas += delta_ms(cur + 10, cur + 11);
			cur += slots_per_iteration_;
		}

		tSolveLRA += delta_ms(cur + 0, cur + 1);
		tCollideSdf += delta_ms(cur + 2, cur + 3);
		tUpdate += delta_ms(cur + 4, cur + 5);
		cur += slots_post_iteration_;
	}

	tCalculateNormals = delta_ms(cur + 0, cur + 1) + delta_ms(cur + 2, cur + 3);

	pass_total_time_ = delta_ms(0, numTimestamp - 1);

	// Sum of every timed kernel (excludes barriers/copies between them; see pass_total_time_).
	float total =
		tWind + tIntegrate + tClearLambdas +
		tHashBuild + tRadixSort + tBuildCell + tBuildNeighbor +
		tSolveStretch + tSolveShear + tSolveBend + tSolveArea + tSolveSelfCollision + tApplyDeltas +
		tSolveLRA + tCollideSdf + tUpdate +
		tCalculateNormals;

	uint32_t c = 0;

	{
		c = 0;
		label_time_[labels_[c++]] = tWind;
		label_time_[labels_[c++]] = tIntegrate;
		label_time_[labels_[c++]] = tClearLambdas;
		label_time_[labels_[c++]] = tHashBuild;
		label_time_[labels_[c++]] = tRadixSort;
		label_time_[labels_[c++]] = tBuildCell;
		label_time_[labels_[c++]] = tBuildNeighbor;
		label_time_[labels_[c++]] = tSolveStretch;
		label_time_[labels_[c++]] = tSolveShear;
		label_time_[labels_[c++]] = tSolveBend;
		label_time_[labels_[c++]] = tSolveArea;
		label_time_[labels_[c++]] = tSolveSelfCollision;
		label_time_[labels_[c++]] = tApplyDeltas;
		label_time_[labels_[c++]] = tSolveLRA;
		label_time_[labels_[c++]] = tCollideSdf;
		label_time_[labels_[c++]] = tUpdate;
		label_time_[labels_[c++]] = tCalculateNormals;
		label_time_[labels_[c++]] = total;
	}

	{
		c = 0;
		label_avg_time_[labels_[c++]] += tWind;
		label_avg_time_[labels_[c++]] += tIntegrate;
		label_avg_time_[labels_[c++]] += tClearLambdas;
		label_avg_time_[labels_[c++]] += tHashBuild;
		label_avg_time_[labels_[c++]] += tRadixSort;
		label_avg_time_[labels_[c++]] += tBuildCell;
		label_avg_time_[labels_[c++]] += tBuildNeighbor;
		label_avg_time_[labels_[c++]] += tSolveStretch;
		label_avg_time_[labels_[c++]] += tSolveShear;
		label_avg_time_[labels_[c++]] += tSolveBend;
		label_avg_time_[labels_[c++]] += tSolveArea;
		label_avg_time_[labels_[c++]] += tSolveSelfCollision;
		label_avg_time_[labels_[c++]] += tApplyDeltas;
		label_avg_time_[labels_[c++]] += tSolveLRA;
		label_avg_time_[labels_[c++]] += tCollideSdf;
		label_avg_time_[labels_[c++]] += tUpdate;
		label_avg_time_[labels_[c++]] += tCalculateNormals;
		label_avg_time_[labels_[c++]] += total;
	}

	time_count_++;

}