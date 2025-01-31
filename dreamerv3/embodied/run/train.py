import re

import embodied
import jax
import numpy as np


def train(agent, env, replay, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(_, __, ___):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            if getattr(replay, "update_visit_count", False):
                replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print("Start training loop.")
    driver._state = None
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()


def ma_train(agents, env, replay, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    for agent in agents:
        timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.MADriver(env, args.num_agents)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset = agents[0].dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(_, __, ___):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agents[0].train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            if getattr(replay, "update_visit_count", False):
                replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            updates.increment()
        if should_sync(updates):
            agents[0].sync()
        if should_log(step):
            agg = metrics.result()
            report = agents[0].report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoints = []
    for i, agent in enumerate(agents):
        if i == 0:
            # For the training agent, its checkpoint will be updated
            checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
            timer.wrap("checkpoint", checkpoint, ["save", "load"])
            checkpoint.step = step
            checkpoint.agent = agent
            checkpoint.replay = replay
            if args.from_checkpoint:
                checkpoint.load(args.from_checkpoint)
            checkpoint.load_or_save()
            should_save(step)  # Register that we jused saved.
            checkpoints.append(checkpoint)
        else:
            # For the other agents, their checkpoint is only loaded and stay frozen
            checkpoint = embodied.Checkpoint()
            checkpoint.agent = agent
            checkpoints.append(checkpoint)
            if args.ma_checkpoint:
                checkpoint.load(args.ma_checkpoint, keys=["agent"])
            else:
                raise ValueError("No MA checkpoint specified.")


    print("Start training loop.")
    # driver._state = None
    policy_train = lambda *args: agents[0].policy(*args, mode="explore" if should_expl(step) else "train")
    policy_frozen = lambda *args: agents[1].policy(*args, mode="eval")
    policy = [policy_train, policy_frozen]
    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoints[0].save()
    logger.write()
