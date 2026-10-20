from resr.runnables.resr_train_common import parse_resr_args, run_resr


if __name__ == "__main__":
    run_resr(
        dataset="toy",
        args=parse_resr_args("Train or evaluate the main ReSR method on Toy."),
    )
