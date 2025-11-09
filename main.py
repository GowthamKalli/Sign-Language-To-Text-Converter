#!/usr/bin/env python
import argparse
import sys
import os
import re
from datetime import datetime

# torchlight
import torchlight
from torchlight import import_class


class StreamLogger:
    """Wrap a stream (stdout/stderr) and optionally copy all writes to a file.

    Additionally looks for lines containing an epoch marker (e.g. 'Epoch 12:' or 'Epoch 12')
    and saves those lines (or blocks) into per-epoch files in a directory if configured.
    """

    EPOCH_RE = re.compile(r"\b[Ee]poch\s*(?:number\s*)?:?\s*(\d+)\b")

    def __init__(self, orig_stream, log_path=None, epoch_dir=None):
        self.orig = orig_stream
        self.log_path = log_path
        self.epoch_dir = epoch_dir
        self._buffer = ""
        if self.log_path:
            # open in append mode
            os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)
            self._log_f = open(self.log_path, 'a', encoding='utf-8')
            header = f"\n--- session start {datetime.now().isoformat()} ---\n"
            self._log_f.write(header)
            self._log_f.flush()
        else:
            self._log_f = None

        if self.epoch_dir:
            os.makedirs(self.epoch_dir, exist_ok=True)

    def write(self, text):
        # write to original stream so user still sees output
        try:
            self.orig.write(text)
        except Exception:
            pass

        # buffer and split into lines so we can detect epoch markers reliably
        self._buffer += text
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._handle_line(line + '\n')

    def _handle_line(self, line):
        # write full line to log file
        if self._log_f:
            try:
                self._log_f.write(line)
                self._log_f.flush()
            except Exception:
                pass

        # check for epoch marker
        m = self.EPOCH_RE.search(line)
        if m and self.epoch_dir:
            epoch_num = int(m.group(1))
            fname = os.path.join(self.epoch_dir, f'epoch_{epoch_num:04d}.txt')
            try:
                # append the single line for now (could be extended to capture blocks)
                with open(fname, 'a', encoding='utf-8') as ef:
                    ts = datetime.now().isoformat()
                    ef.write(f'[{ts}] ' + line)
            except Exception:
                pass

    def flush(self):
        try:
            if self._log_f:
                self._log_f.flush()
        except Exception:
            pass
        try:
            self.orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # add global logging options
    parser.add_argument('--metrics-log', help='Path to append all stdout/stderr to this log file (e.g. logs/metrics_log.txt)', default=None)
    parser.add_argument('--epoch-dir', help='Directory to write per-epoch metric lines (e.g. logs/epochs/). If provided, any line matching "Epoch <num>" will be appended into epoch_<num>.txt', default=None)

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        # integrate subparser of each processor so their specific args still work
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # If user requested metrics logging, wrap stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logger_stdout = None
    logger_stderr = None
    try:
        if arg.metrics_log or arg.epoch_dir:
            # use same file for both stdout and stderr unless user configures otherwise
            logger_stdout = StreamLogger(original_stdout, log_path=arg.metrics_log, epoch_dir=arg.epoch_dir)
            logger_stderr = StreamLogger(original_stderr, log_path=arg.metrics_log, epoch_dir=arg.epoch_dir)
            sys.stdout = logger_stdout
            sys.stderr = logger_stderr

        # start
        if not arg.processor:
            parser.print_help()
            sys.exit(1)

        Processor = processors[arg.processor]
        # instantiate processor with the remaining args (skip the script and processor name)
        # NOTE: many processors expect argv-like list — keep previous behaviour
        p = Processor(sys.argv[2:])

        p.start()

    finally:
        # restore streams so we close files cleanly
        try:
            if logger_stdout and hasattr(logger_stdout, '_log_f') and logger_stdout._log_f:
                logger_stdout._log_f.write(f"\n--- session end {datetime.now().isoformat()} ---\n")
                logger_stdout._log_f.close()
        except Exception:
            pass
        try:
            if logger_stderr and hasattr(logger_stderr, '_log_f') and logger_stderr._log_f:
                logger_stderr._log_f.close()
        except Exception:
            pass

        sys.stdout = original_stdout
        sys.stderr = original_stderr

