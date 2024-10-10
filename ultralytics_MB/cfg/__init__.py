# ultralytics_MB YOLO 🚀, AGPL-3.0 license

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics_MB.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    IS_VSCODE,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_FILE,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    checks,
    colorstr,
    deprecation_warn,
    vscode_msg,
    yaml_load,
    yaml_print,
)

# Define valid tasks and modes
MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "classify": "yolo11n-cls.pt",
    "pose": "yolo11n-pose.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}

CLI_HELP_MSG = f"""
    Arguments received: {str(['yolo'] + sys.argv[1:])}. ultralytics_MB 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics_MB.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

    4. Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    6. Explore your datasets using semantic search and SQL with a simple GUI powered by ultralytics_MB Explorer API
        yolo explorer

    5. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics_MB.com
    Community: https://community.ultralytics_MB.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Define keys for arg type checks
CFG_FLOAT_KEYS = {"warmup_epochs", "box", "cls", "dfl", "degrees", "shear", "time"}
CFG_FRACTION_KEYS = {
    "dropout",
    "iou",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}  # fraction floats 0.0 - 1.0
CFG_INT_KEYS = {
    "epochs",
    "patience",
    "batch",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "workspace",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data.
        overrides (str | Dict | optional): Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from ultralytics_MB.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg, hard=True):
    """Check ultralytics_MB configuration argument types and values."""
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)


def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics_MB.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)


def _handle_deprecation(custom):
    """Hardcoded function to handle deprecated config keys."""

    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")

    return custom


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list. If
    any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Args:
        custom (dict): a dictionary of custom configuration options
        base (dict): a dictionary of base configuration options
        e (Error, optional): An optional error that is passed by the calling function.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings. The function considers cases where the first
    argument ends with '=' or the second starts with '=', as well as when the middle one is an equals sign.

    Args:
        args (List[str]): A list of strings where each element is an argument.

    Returns:
        (List[str]): A list of strings where the arguments around isolated '=' are merged.
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            del args[i + 1]
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            del args[i + 1]
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args


def handle_yolo_hub(args: List[str]) -> None:
    """
    Handle ultralytics_MB HUB command-line interface (CLI) commands.

    This function processes ultralytics_MB HUB CLI commands such as login and logout.
    It should be called when executing a script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments

    Example:
        ```bash
        python my_script.py hub login your_api_key
        ```
    """
    from ultralytics_MB import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""
        # Log in to ultralytics_MB HUB using the provided API key
        hub.login(key)
    elif args[0] == "logout":
        # Log out from ultralytics_MB HUB
        hub.logout()


def handle_yolo_settings(args: List[str]) -> None:
    """
    Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset.
    It should be called when executing a script with arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Examples:
        >>> handle_yolo_settings(["reset"])  # Reset YOLO settings
        >>> handle_yolo_settings(["default_cfg_path=yolo11n.yaml"])  # Update a specific setting

    Notes:
        - If no arguments are provided, the function will display the current settings.
        - The 'reset' command will delete the existing settings file and create new default settings.
        - Other arguments are treated as key-value pairs to update specific settings.
        - The function will check for alignment between the provided settings and the existing ones.
        - After processing, the updated settings will be displayed.
        - For more information on handling YOLO settings, visit:
          https://docs.ultralytics_MB.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics_MB.com/quickstart/#ultralytics-settings"  # help URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info("Settings reset successfully")  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        print(SETTINGS)  # print the current settings
        LOGGER.info(f"💡 Learn more about ultralytics_MB Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")


def handle_explorer(args: List[str]):
    """
    Launches a graphical user interface that provides tools for interacting with and analyzing datasets using the
    ultralytics_MB Explorer API. It checks for the required 'streamlit' package and informs the user that the Explorer
    dashboard is loading.

    Args:
        args (List[str]): A list of optional command line arguments.

    Examples:
        ```bash
        yolo explorer data=data.yaml model=yolo11n.pt
        ```

    Notes:
        - Requires 'streamlit' package version 1.29.0 or higher.
        - The function does not take any arguments or return any values.
        - It is typically called from the command line interface using the 'yolo explorer' command.
    """
    checks.check_requirements("streamlit>=1.29.0")
    LOGGER.info("💡 Loading Explorer dashboard...")
    subprocess.run(["streamlit", "run", ROOT / "data/explorer/gui/dash.py", "--server.maxMessageSize", "2048"])



def parse_key_value_pair(pair: str = "key=value"):
    """
    Parses a key-value pair string into separate key and value components.

    Args:
        pair (str): A string containing a key-value pair in the format "key=value".

    Returns:
        (tuple): A tuple containing two elements:
            - key (str): The parsed key.
            - value (str): The parsed value.

    Raises:
        AssertionError: If the value is missing or empty.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolo11n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo11n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    Notes:
        - The function splits the input string on the first '=' character.
        - Leading and trailing whitespace is removed from both key and value.
        - An assertion error is raised if the value is empty after stripping.
    """
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """Convert a string to an underlying type such as int, float, bool, etc."""
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return eval(v)
        except:  # noqa E722
            return v


def entrypoint(debug=""):
    """
    ultralytics_MB entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the ultralytics_MB CLI, parsing command-line arguments and
    executing the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        >>> entrypoint("predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - If no arguments are passed, the function will display the usage help message.
        - For a list of all available commands and their arguments, see the provided help messages and the
          ultralytics_MB documentation at https://docs.ultralytics_MB.com.
    """
    args = (debug.split(" ") if debug else sys.argv)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: yaml_print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "logout": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "explorer": lambda: handle_explorer(),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 'mode' argument is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"WARNING ⚠️ 'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics_MB import RTDETR

        model = RTDETR(model)  # no task argument
    elif "fastsam" in stem:
        from ultralytics_MB import FastSAM

        model = FastSAM(model)
    elif "sam2" in stem:
        from ultralytics_MB import SAM2

        model = SAM2(model)
    elif "sam" in stem:
        from ultralytics_MB import SAM

        model = SAM(model)
    else:
        from ultralytics_MB import YOLO

        model = YOLO(model, task=task)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING ⚠️ conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in ("predict", "track") and "source" not in overrides:
        overrides["source"] = DEFAULT_CFG.source or ASSETS
        LOGGER.warning(f"WARNING ⚠️ 'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in ("train", "val"):
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics_MB.com/modes/{mode}")

    # Recommend VS Code extension
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """Copy and create a new default configuration file with '_copy' appended to its name."""
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolo11n.pt')
    entrypoint(debug="")
