"""Replay-based SuperMimic collection/model/imagination training loop."""

from dataclasses import replace
from secrets import token_hex
from time import perf_counter

import torch
from setproctitle import setproctitle

from newrl.commands import with_used_observations
from newrl.commands.train import EpisodeStats, TrainConfig, _make_logger, _save_checkpoint, _task_output_dir
from newrl.env import StatefulEnvironment
from newrl.utils.memory import CudaMemoryProfiler, storage_bytes, write_memory_chart


def train_supermimic(cfg: TrainConfig) -> None:
    cfg = replace(cfg, task=cfg.algo.prepare_task(cfg.task))
    torch.manual_seed(cfg.seed)
    cfg.task.mdp.seed = cfg.seed
    title = f'{cfg.task.name}-{cfg.algo.name}-ETA:'
    setproctitle(f'{title}--:--:--')
    profiler = CudaMemoryProfiler(torch.device(cfg.task.device)) if cfg.memory_profile_iters else None
    if profiler is not None:
        profiler.begin_phase()
    task = with_used_observations(cfg.task, cfg.algo) if cfg.discard_unused_obs else cfg.task
    env = StatefulEnvironment(task)
    learner = cfg.algo.build(env)
    memory = profiler.end_phase('env_build') if profiler is not None else {}
    samples = []
    profile_end = min(cfg.iterations, cfg.memory_profile_warmup + cfg.memory_profile_iters)
    if cfg.compile_policy:
        learner.act = torch.compile(learner.act, fullgraph=True)
    run_id = token_hex(4)
    logger = _make_logger(cfg, run_id=run_id)
    stats = EpisodeStats(env.mdp.rewards)
    frames = 0
    directory = _task_output_dir(cfg.task.name) / 'checkpoints' / run_id
    try:
        batch, carry = learner.collect(env, cfg.algo.warmup_steps)
        frames += batch.numel()
        stats.update(batch)
        start = perf_counter()
        for iteration in range(cfg.iterations):
            sample = profiler is not None and (iteration == 0 or cfg.memory_profile_warmup <= iteration < profile_end)
            if sample:
                profiler.begin_phase()
            begin = perf_counter()
            batch, carry = learner.collect(env, cfg.algo.update_every, carry)
            torch.cuda.synchronize(env.device)
            collection_end = perf_counter()
            metrics = stats.update(batch)
            if sample:
                memory.update(profiler.end_phase('rollout'))
                memory.update(profiler.rollout(batch, carry))
                profiler.begin_phase()
            metrics.update(learner.update_world_model())
            model_end = perf_counter()
            metrics.update(learner.update_actor_critic())
            end = perf_counter()
            if sample:
                memory.update(profiler.end_phase('update'))
                memory.update(profiler.environment(env))
                memory.update(profiler.learner(learner))
                memory['memory/buffers/replay_mib'] = storage_bytes(learner.replay.storage) / 2**20
                memory['memory/buffers/optimizer_mib'] = storage_bytes(*(optimizer.state_dict() for optimizer in (learner.opt_wm, learner.opt_actor, learner.opt_critic))) / 2**20
                memory['memory/warmup'] = int(iteration < cfg.memory_profile_warmup)
                metrics.update(memory)
                samples.append(memory.copy())
                memory.clear()
            frames += batch.numel()
            metrics.update({
                'env_frames':frames, 'replay/rows':learner.replay.filled,
                'performance/rollout_time':collection_end - begin,
                'performance/rollout_fps':batch.numel() / (collection_end - begin),
                'performance/world_model_time':model_end - collection_end,
                'performance/policy_critic_time':end - model_end,
                'performance/iter_time':end - begin,
            })
            logger.log(metrics, step=iteration)
            summary = ' '.join(f'{key}={value:.5g}' for key,value in metrics.items())
            print(f'iter={iteration} {summary}', flush=True)
            completed = iteration + 1
            if samples and completed == profile_end:
                path = _task_output_dir(cfg.task.name) / 'profiles' / run_id / 'gpu_memory.svg'
                write_memory_chart(path, samples)
            if cfg.checkpoint_interval and (completed % cfg.checkpoint_interval == 0 or completed == cfg.iterations):
                path = _save_checkpoint(directory, learner, cfg, completed, frames)
                logger.save_checkpoint(path)
                print(f'Saved checkpoint {path}', flush=True)
            remaining = round((perf_counter() - start) / completed * (cfg.iterations - completed))
            hours, remaining = divmod(remaining, 3600)
            minutes, seconds = divmod(remaining, 60)
            setproctitle(f'{title}{hours:02d}:{minutes:02d}:{seconds:02d}')
    finally:
        logger.finish()
