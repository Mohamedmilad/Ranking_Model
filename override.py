from pytorch_lightning.trainer.connectors.logger_connector import _LoggerConnector
import pytorch_lightning as pl
from typing import Any, Iterable, Optional, Union

from pytorch_lightning.trainer.connectors.logger_connector.result import _METRICS, _OUT_DICT, _PBAR_DICT
def train_step_override(self):
    if self.trainer.fit_loop._should_accumulate() and self.trainer.lightning_module.automatic_optimization:
            return
    self.c=self.c+1
    loss_tensor = (self.metrics["log"])['train_loss_step']
    loss_value = loss_tensor.item()
    self.writer.add_scalar('loss',loss_value,self.c)
    self.writer.flush()
    # when metrics should be logged
    assert isinstance(self._first_loop_iter, bool)
    if self.should_update_logs or self.trainer.fast_dev_run:
        print(self.metrics["log"])
        self.log_metrics(self.metrics["log"])
def __init__(self, trainer: "pl.Trainer") -> None:
        self.c=0
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter('runs/experiment_1')
        self.trainer = trainer
        self._progress_bar_metrics: _PBAR_DICT = {}
        self._logged_metrics: _OUT_DICT = {}
        self._callback_metrics: _OUT_DICT = {}
        self._current_fx: Optional[str] = None
        # None: hasn't started, True: first loop iteration, False: subsequent iterations
        self._first_loop_iter: Optional[bool] = None
_LoggerConnector.update_train_step_metrics=train_step_override
_LoggerConnector.__init__=__init__