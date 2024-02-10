#!/usr/bin/python3
import argparse
import logging
import os
import time
from typing import Tuple, List, Union

import torch
import configs
import console_logger
import log_helper_wrapper


class Timer:
    time_measure = 0

    def tic(self): self.time_measure = time.time()

    def toc(self): self.time_measure = time.time() - self.time_measure

    @property
    def diff_time(self): return self.time_measure

    @property
    def diff_time_str(self): return str(self)

    def __str__(self): return f"{self.time_measure:.4f}s"

    def __repr__(self): return str(self)


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description='PyTorch Maximals radiation setup', add_help=True)
    parser.add_argument('--iterations', default=int(1e12), help="Iterations to run forever", type=int)
    parser.add_argument('--generate', default=False, action="store_true", help="To generate the gold")
    parser.add_argument('--gold_file', default="./gold.matrix", help="Path to Golden")
    parser.add_argument('--alpha', default=1.0, help="Alpha value")
    parser.add_argument('--beta', default=0, help="Beta value")
    parser.add_argument('--log_interval', default=1, help="frequency of #IT logging", type=int)
    parser.add_argument('--size', default=1024, help="matrix size", type=int)

    parser.add_argument('--verbose', default=False, action="store_true", help="For verbose console")
    args = parser.parse_args()

    # Check if it is only to generate the gold values
    if args.generate is True:
        args.iterations = 1

    args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
    return args, args_text_list


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: Union[None, float]) -> bool:
    """ Compare based or not in a threshold, if the threshold is none then it is equal comparison    """
    if threshold is not None:
        return bool(
            torch.all(
                torch.le(
                    torch.abs(
                        torch.subtract(rhs, lhs)
                    ), threshold
                )
            )
        )
    else:
        return bool(torch.equal(rhs, lhs))


def describe_error(input_tensor: torch.tensor) -> Tuple[int, int, float, float]:
    flattened_tensor = input_tensor.flatten()
    is_nan_tensor, is_inf_tensor = torch.isnan(flattened_tensor), torch.isinf(flattened_tensor)
    has_nan, has_inf = int(torch.any(is_nan_tensor)), int(torch.any(is_inf_tensor))
    filtered_tensor = flattened_tensor[~is_nan_tensor & ~is_inf_tensor]
    min_val = float(torch.min(filtered_tensor)) if filtered_tensor.numel() > 0 else 0
    max_val = float(torch.max(filtered_tensor)) if filtered_tensor.numel() > 0 else 0
    return has_nan, has_inf, min_val, max_val


def compare_output(output_tensor: torch.tensor, golden_tensor: torch.tensor, output_logger: logging.Logger) -> int:
    output_errors = 0

    # Get non-equal elements' indices
    # Identify non-equal elements
    diff_mask = torch.ne(output_tensor, golden_tensor)
    # Get indices where elements differ
    diff_indices = torch.nonzero(diff_mask)
    for index in diff_indices:
        i, j = index
        gold_value = golden_tensor[i, j]
        read_value = output_tensor[i, j]

        if gold_value != read_value:
            error_detail = f"p:[{i}, {j}] r:{read_value}, e:{gold_value}"
            if output_logger and output_errors < 10:
                output_logger.debug(error_detail)

            log_helper_wrapper.log_error_detail(error_detail)
            output_errors += 1

    # ------------ Check error on the whole output -------------------------------------------------------------
    # Not necessary to save everything, only the good info
    # Data on output tensor
    has_nan, has_inf, min_val, max_val = describe_error(input_tensor=output_tensor)
    error_detail_out = f"output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
    # Data on abs differences
    abs_diff = torch.abs(torch.subtract(output_tensor, golden_tensor))
    has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = describe_error(input_tensor=abs_diff)
    error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
    output_errors += 1
    if output_logger:
        output_logger.error(error_detail_out)
    log_helper_wrapper.log_error_detail(error_detail_out)

    return output_errors


def compare(output_tensor: torch.tensor, golden_tensor: torch.tensor, output_logger: logging.Logger,
            float_threshold: float, ) -> int:
    # global TEST
    # TEST += 1
    # if TEST == 3:
    #     output_tensor[3, 6] = 39304

    # Make sure that they are on CPU
    out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    if out_is_cuda or golden_is_cuda:
        log_helper_wrapper.log_and_crash(
            fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

    # First check if the tensors are equal or not
    if equal(lhs=output_tensor, rhs=golden_tensor, threshold=float_threshold) is True:
        return 0

    # ------------ Check the size of the tensors
    if output_tensor.shape != golden_tensor.shape:
        info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
        if output_logger:
            output_logger.error(info_detail)
        log_helper_wrapper.log_info_detail(info_detail)

    # ------------ Main check
    output_errors = compare_output(output_tensor=output_tensor, golden_tensor=golden_tensor,
                                   output_logger=output_logger)

    # ------------ log and return
    if output_errors != 0:
        log_helper_wrapper.log_error_count(error_count=output_errors)
    return output_errors


def check_and_setup_gpu() -> None:
    # Disable all torch grads
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        log_helper_wrapper.log_and_crash(fatal_string=f"Device {configs.DEVICE} not available.")
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        log_helper_wrapper.log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")


@torch.no_grad()
def main():
    args, args_text_list = parse_args()
    # Define DNN goal
    log_helper_wrapper.start_setup_log_file(framework_name="SGEMMPytorch", torch_version=torch.__version__,
                                            gpu=torch.cuda.get_device_name(),
                                            args_conf=args_text_list, activate_logging=not args.generate,
                                            float_threshold=configs.FLOAT_ERROR_THRESHOLD,
                                            log_print_interval=args.log_interval)

    # Check if a device is ok and disable grad
    check_and_setup_gpu()

    # Defining a timer
    timer = Timer()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.verbose is True else None

    # Load if it is not a gold generating op
    timer.tic()
    # This will save time
    if args.generate is False:
        # Save everything in the same list
        golden, input_a, input_b = torch.load(args.gold_file)
        input_a, input_b = input_a.to(configs.DEVICE), input_b.to(configs.DEVICE)
    else:
        r1, r2 = configs.GENERATOR_MIN_ABS_VALUE_GEMM, configs.GENERATOR_MAX_ABS_VALUE_GEMM
        input_a = torch.FloatTensor(args.size, args.size).uniform_(r1, r2).to(configs.DEVICE)
        input_b = torch.FloatTensor(args.size, args.size).uniform_(r1, r2).to(configs.DEVICE)
        golden = torch.Tensor()

    timer.toc()
    golden_load_diff_time = timer.diff_time_str

    if terminal_logger:
        terminal_logger.debug("\n".join(args_text_list))
        terminal_logger.debug(f"Time necessary to load the golden outputs, model, and inputs: {golden_load_diff_time}")

    # Main setup loop
    for setup_iteration in range(args.iterations):
        timer.tic()
        log_helper_wrapper.start_iteration()
        gemm_output = torch.matmul(input_a, input_b)
        torch.cuda.synchronize(device=configs.DEVICE)
        log_helper_wrapper.end_iteration()
        timer.toc()
        kernel_time = timer.diff_time
        # Always copy to CPU
        timer.tic()
        gemm_output_cpu = gemm_output.to(configs.CPU)
        timer.toc()
        copy_to_cpu_time = timer.diff_time
        # Then compare the golden with the output
        timer.tic()
        errors = 0
        if args.generate is False:
            errors = compare(output_tensor=gemm_output_cpu, golden_tensor=golden, output_logger=terminal_logger,
                             float_threshold=configs.FLOAT_ERROR_THRESHOLD)
        else:
            golden = gemm_output_cpu

        timer.toc()
        comparison_time = timer.diff_time

        # Reload all the memories after error
        if errors != 0:
            if terminal_logger:
                terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
            del input_a
            del input_b
            del gemm_output
            # Free cuda memory
            torch.cuda.empty_cache()
            # Everything in the same list
            golden, input_a, input_b = torch.load(args.gold_file)
            input_a, input_b = input_a.to(configs.DEVICE), input_b.to(configs.DEVICE)

        # Printing timing information
        if terminal_logger:
            wasted_time = comparison_time + copy_to_cpu_time
            time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
            iteration_out = f"It:{setup_iteration:<3} kernel time:{kernel_time:.5f}, "
            iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
            iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
            terminal_logger.debug(iteration_out)

    if args.generate is True:
        final_golden = [golden, input_a, input_b]
        torch.save(final_golden, args.gold_file)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    log_helper_wrapper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        log_helper_wrapper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
