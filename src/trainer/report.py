"""Metrics aggregation, JSON/NPZ serialization, loss chart rendering."""

import json
import os

from trainer.epoch import TestEpoch
from utils.utils import check_dir, draw_loss_line


class MetricsReporter:
    """Collects training/validation metrics and writes artifacts to disk."""

    def __init__(self, log_dir, mylogger):
        self.log_dir = log_dir
        self.mylogger = mylogger
        self.train_loss_line = {"x": [], "y": []}
        self.val_loss_line = {"x": [], "y": []}

    def record_train(self, step_id, loss):
        self.train_loss_line["x"].append(step_id)
        self.train_loss_line["y"].append(loss)

    def record_val(self, step_id, loss):
        self.val_loss_line["x"].append(step_id)
        self.val_loss_line["y"].append(loss)

    def save_metrics_json(self, full_metrics):
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(full_metrics, f, indent=2, default=str)
        self.mylogger.info(f"[Metrics] saved to {metrics_path}")

    def save_npz_results(self, data_bundles, model, aux_pool_sets_by_key, args):
        """Save NPZ prediction results for each bundle."""
        for bundle in data_bundles:
            model.set_graph(bundle["key"])
            model.set_aux_pools(aux_pool_sets_by_key[bundle["key"]][0])
            bundle_log_dir = (
                os.path.join(self.log_dir, bundle["key"])
                if len(data_bundles) > 1
                else self.log_dir
            )
            check_dir(bundle_log_dir, mkdir=True)
            TestEpoch(
                bundle["test_loader"],
                model,
                args,
                log_dir=bundle_log_dir,
                save=True,
                resample=False,
                full_nodes=True,
            )

    def draw_chart(self):
        draw_loss_line(
            self.train_loss_line,
            self.val_loss_line,
            os.path.join(self.log_dir, "loss.png"),
        )
