from aibro.inference import Inference

DRYRUN = False

if DRYRUN:
    # test the model repo by dry run
    Inference.deploy(
        artifacts_path = "./alpaca_model_repo",
        dryrun = True,
    )
else:
    # create an inference API
    Inference.deploy(
        model_name = "alpaca_model",
        machine_id_config = "c5.large.od",
        artifacts_path = "./alpaca_model_repo",
    )