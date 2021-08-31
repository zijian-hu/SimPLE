import sys

from .logger import Logger

# for type hint
from typing import Any, Dict, Optional, Union
from argparse import Namespace

from torch.nn import Module


class PrintLogger(Logger):
    def __init__(self,
                 log_dir: str,
                 config: Union[Namespace, Dict[str, Any]],
                 is_display_plots: bool = False):
        super().__init__(
            log_dir=log_dir,
            config=config,
            log_info_key_map={
                "top1_acc": "mean_acc",
                "top5_acc": "mean_top5_acc",
            })

        self.is_display_plots = is_display_plots

    def log(self,
            log_info: Dict[str, Any],
            step: Optional[int] = None,
            file=sys.stdout,
            flush: bool = False,
            sep: str = "\n\t",
            prefix: Optional[str] = None,
            log_info_override: Optional[Dict[str, Any]] = None,
            **kwargs):
        # process log_info
        log_info = self.process_log_info(log_info, prefix=prefix, log_info_override=log_info_override)

        if len(log_info) == 0:
            return

        log_info, plot_info = self.separate_plot(log_info)

        if self.is_display_plots:
            # display plots
            for key, fig in plot_info.items():
                if hasattr(fig, "show"):
                    fig.show()

        output_str = sep.join(f"{str(key)}: {str(val)}" for key, val in log_info.items())
        if step is not None:
            print(f"Step {step}:\n\t{output_str}", file=file, flush=flush)
        else:
            print(output_str, file=file, flush=flush)

        # invoke all log hook functions
        self.call_log_hooks(log_info)

    def watch(self, model: Module, **kwargs):
        # TODO: implement
        pass

    def save(self, output_path: str):
        # TODO: implement
        pass
