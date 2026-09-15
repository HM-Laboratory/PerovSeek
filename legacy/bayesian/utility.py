import torch
from torch.multiprocessing import Pool
from functools import partial
from botorch.optim import optimize_acqf
def single_restart_optimization(acq_function, bounds, q, raw_samples, options, equality_constraints, inequality_constraints,restart_idx, device):
    # 每个重启独立运行，设置不同的随机种子
    torch.manual_seed(restart_idx)
    # 生成初始点
    raw_candidates = torch.rand(raw_samples, bounds.shape[-1], device=device) * (bounds[1] - bounds[0]) + bounds[0]
    # 单次优化
    candidate, value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=1,  # 单次重启
        raw_samples=raw_samples,
        options=options,
        equality_constraints = equality_constraints,
        inequality_constraints = inequality_constraints,
        return_best_only=False  # 返回所有结果以供后续筛选
    )

    # 转移到 CPU 并清理显存
    candidate, value = candidate.cpu(), value.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # 清理显存
    return candidate, value


def parallel_optimize_acqf(num_processes,acq_function, bounds, q, num_restarts, raw_samples, options,equality_constraints,inequality_constraints, device):
    # 创建并行池

    with Pool(processes=num_processes) as pool:
        # 使用 partial 固定参数
        worker = partial(single_restart_optimization, acq_function, bounds, q, raw_samples, options,equality_constraints,inequality_constraints,device=device)
        # 并行执行多个重启
        results = pool.map(worker, range(num_restarts))

    # 只保留最佳结果，而不是所有结果
    best_candidate, best_value = None, float('-inf')
    for candidate, value in results:
        if value.max() > best_value:
            best_value = value.max()
            best_candidate = candidate
    
    return best_candidate, best_value
