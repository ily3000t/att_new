import json
from pathlib import Path

from amb.envs.base_logger import BaseLogger
from amb.envs.network.cacc_metrics import CACCEpisodeMetrics


class NetworkLogger(BaseLogger):

    def get_task_name(self):
        return self.env_args["scenario"]

    def eval_init(self, n_eval_rollout_threads):
        super().eval_init(n_eval_rollout_threads)
        if self.task_name not in ("catchup", "slowdown"):
            return
        self.cacc_eval_index = getattr(self, "cacc_eval_index", 0) + 1
        self.cacc_eval_path = Path(self.run_dir) / f"cacc_eval_{self.cacc_eval_index:04d}.jsonl"
        self.cacc_threshold = self.env_args.get("ttc_threshold_s", 1.0)
        self.cacc_running = [CACCEpisodeMetrics(self.cacc_threshold) for _ in range(n_eval_rollout_threads)]
        self.cacc_episodes = []

    def eval_per_step(self, eval_data, rewards_title="eval_per_step_rewards"):
        super().eval_per_step(eval_data, rewards_title)
        if self.task_name not in ("catchup", "slowdown"):
            return
        self.cacc_eval_label = rewards_title
        rewards, infos = eval_data[2], eval_data[4]
        for thread_id in range(self.n_eval_rollout_threads):
            self.cacc_running[thread_id].update(infos[thread_id], rewards[thread_id])

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        if self.task_name not in ("catchup", "slowdown"):
            return
        record = self.cacc_running[tid].result()
        record.update({"worker": tid, "evaluation": self.cacc_eval_label})
        self.cacc_episodes.append(record)
        with self.cacc_eval_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.cacc_running[tid] = CACCEpisodeMetrics(self.cacc_threshold)
