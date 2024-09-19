from pytorch_lightning.trainer.connectors.logger_connector import _LoggerConnector
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
_LoggerConnector.update_train_step_metrics=train_step_override