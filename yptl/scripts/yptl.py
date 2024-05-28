import argparse  # noqa: D100

from yptl.utilities.yaml import (
    YPTLConfig,
    create_datamodule_from_config,
    create_model_from_yaml_config,
    create_trainer_from_config,
)


def parse_args() -> argparse.Namespace:  # noqa: D103
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "template",
        type=str,
        help="yptl configuration file in yaml format",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Change logging level to debug",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run yptl script.

    Create model, datamodule and trainer objects based on template file.
    Run training and test.
    """
    args = parse_args()

    yptl_config = YPTLConfig.from_yaml(args.template)
    model = create_model_from_yaml_config(yptl_config.model)
    datamodule = create_datamodule_from_config(yptl_config.datamodule)
    trainer = create_trainer_from_config(yptl_config.trainer)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
