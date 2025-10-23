# SeCoNoHe (SET's ComfyUI Node Helpers)

[Full docs](https://set-soft.github.io/seconohe/seconohe.html)

I have a few ComfyUI custom nodes that I wrote, and I started to repeat the same `utils` over and over on each node.
Soon realized it was a waste of resources and a change or fix in one of the `utils` was hard to apply to all the nodes.
So I separated them as a Python module available from PyPi.

So this is the Python module. The functionality found here has just one thing in common: was designed to be used by ComfyUI nodes.
Other than that you'll find the functionality is really heterogeneous.

## &#x0001F4D6; Table of Contents

- &#x0001F3AF; [Core Goals](#-core-goals)
- &#x26A0;&#xFE0F; [Important Remarks for Developers](#&#xFE0F;-important-remarks-for-developers)
  - [Relative Imports](#relative-imports)
  - [Recommended Directory Structure](#recommended-directory-structure)
- &#x2728; [Included Functionality](#-included-functionality)
  - &#x0001F50A; [Logger](#-logger)
  - &#x0001F35E; [ComfyUI Toast Notifications](#-comfyui-toast-notifications)
  - &#x0001F4BE; [File Downloader](#-file-downloader)
  - &#x270D;&#xFE0F; [Automatic Node Registration](#&#xFE0F;-automatic-node-registration)
  - &#x2699;&#xFE0F; [PyTorch Helpers](#&#xFE0F;-pytorch-helpers)
  - &#x0001F39B;&#xFE0F; [Changing Widget Values](#&#xFE0F;-changing-widget-values)
  - [Batch iterator](#batch-iterator)
  - [Color parser](#color-parser)
  - [Foreground estimation](#foreground-estimation)
- &#x0001F680; [Examples of Nodes Using SeCoNoHe](#-examples-of-nodes-using-seconohe)
- &#x0001F4DC; [Project History](#-project-history)
- &#x2696;&#xFE0F; [License](#&#xFE0F;-license)
- &#x0001F64F; [Attributions](#-attributions)


## &#x0001F3AF; Core Goals

One of the main goals of these helpers is to be **lightweight and non-intrusive**.

- **Minimal Dependencies:** The library aims to pull in as few new dependencies as possible, ideally relying only on what's already
  available in a standard ComfyUI installation.
- **Graceful Fallbacks:** For optional features, the helpers check for libraries like `requests` or `colorama` and fall back to built-in
  Python functionality if they are not found. This prevents SeCoNoHe from being the cause of installation conflicts.
- **Safe PyTorch Handling:** PyTorch (`torch`, `torchaudio`) is explicitly *not* listed as a package dependency to avoid `pip` installing
  a duplicate, multi-gigabyte copy. The helpers assume PyTorch is already present in the ComfyUI environment.

More on the fallbacks:

- `requests`: I'm quite sure that all ComfyUI installs has it installed, but isn't explicitly listed in the ComfyUI dependencies.
  If not available we use `urllib`, which is part of the Python core.
  I think, and have no real proof, that `requests` is more robust than `urllib`.
  For this reason if `requests` is installed the download code will use it.
- `colorama`: The logger code tries to use colors for DEBUG, WARNING and ERROR. If `colorama` is installed we use the colors from it,
  otherwise we use the classic ANSI escape sequences. I found a lot of ComfyUI nodes that just uses the ANSI sequences, but I just guess
  that using `colorama` is more robust.

In the list of dependencies I included TQDM (progress bar), which is redundant because ComfyUI depends on it.


## &#x26A0;&#xFE0F; Important Remarks for Developers

Here are some things I learned the hard way that might be useful for other ComfyUI node creators.


### Relative Imports

- &#x2705; **ALWAYS** use relative imports for code *within* your node package (e.g., `from .utils import my_helper`).
- &#x26D4; **NEVER** use absolute imports for your own node's files.
- &#x0001F6AB; **NEVER** modify `sys.path` in any code that ComfyUI will import. This is a common cause of mysterious bugs and conflicts with other custom nodes.
- **ONLY** use absolute imports for dependencies and ComfyUI functionality.

For this use a very isolated version of the Python `src` directory structure recommendation.

### Recommended Directory Structure

Using what Python calls `src` structure is strongly recommended, why? because most modern Python tools assume this is the case,
and most of them misserably fail if you don't use it.

Here is what I recommend:

```
root
 |
 \-- __init__.py      <-- Your nodes registration, the only Python in your root
 \-- pyproject.toml   <-- Nodes and project information, the modern way, and needed by ComfyUI registry
 |
 \-- src/             <-- The `src` magic name
 |    |
 |    \-- nodes/      <-- An extra level of indirection, helps with the relative vs absolute imports
 |    |    |
 |    |    \-- __init__.py    <-- Internal initialization, __version__, NODES_NAME, main_logger, etc. here
 |    |    \-- nodes.py       <-- One or more `nodes_xxx.py` files containing the implementation of your nodes
 |    |    |
 |    |    \-- utils/         <-- One or more submodules with stuff you use in your nodes
 |    |         |
 |    |         \-- __init__.py    <-- Always put an init inside them, usually empty
 |    |
 |    \-- tests/                   <-- Regression tests can be put here
 |         |
 |         \-- bootstrap/
 |         |    |
 |         |    \-- __init__.py    <-- A `sys.path` nasty trick used for the regression tests
 |         |
 |         \-- test_xxx.py          <-- Group of regression tests
 |
 \-- tool/  <-- Command line tools that uses functionality shared with your nodes
      |
      \-- bootstrap/
      |    |
      |    \-- __init__.py    <-- A `sys.path` nasty trick used for the tools
      |
      \-- xxxx.py    <-- A tool
```

Of course `tool` and `tests` are optional. If you don't have them you can save the extra `nodes` level.
But using the extra level allows to add them easily in the future.

The [bootstrap](https://github.com/set-soft/AudioSeparation/blob/main/tool/bootstrap/__init__.py) is used to allow absolute imports
in the command line tools and regression tests.

In the case of the tools you just use things like this:

```python
import bootstrap  # noqa: F401
from src.nodes import main_logger
from src.nodes.db.hash import get_hash
from src.nodes.db.models_db import load_known_models, save_known_models, get_db_filename
from src.nodes.utils.misc import cli_add_verbose
```

Note that here we pretend `src` is a module. As we insert the correct path the code in `tool` will import your `src`.
This is ok because they are standalone tools. **NEVER** modify the `sys.path` in the code that will be imported by ComfyUI,
I saw a lot of nodes doing it.

In the case of the regression tests the imports looks like this:

```python
import bootstrap  # noqa: F401
from nodes.nodes_audio import AudioBatch
```

Here we pretend that `nodes` is the package.

Note that this mechanism allows the use of:

- Relative imports in all the code that belongs to your node, this includes what the above example shows as `src/nodes/utils`.
  In the example `src/nodes/nodes.py` will do `from .utils import xxxx` or `from .utils.yyyy import xxxx`
- Absolute imports from `tests` and `tool`, avoiding the classic error `attempted relative import with no known parent package`

Using this you won't get:

- Errors like `attempted relative import beyond top-level package`
- Mysterious problems because you imported a module named `utils` from the ComfyUI core, or from another node. In particular
  after changing the order of the imports, or doing a "non-top-level" import.


## &#x2728; Included Functionality

Here you'll find an explanation of the functionality.

A full reference can be found [here](https://set-soft.github.io/seconohe/seconohe.html).


### &#x0001F50A; Logger

The ComfyUI console logs are a nightmare, and I don't want to make it worst, so I use a logger that:

- **Clear Prefixing:** All messages are prefixed with your chosen `NODES_NAME` (e.g., `[MyAwesomeNode] Inference complete.`),
  so you always know which custom node is talking.
- **Smart Coloring:** Informational messages are kept clean (no color), while `DEBUG`, `WARNING`, and `ERROR` messages are colored to
  stand out. Text labels are included if colors fail.
- **Browser Notifications:** Warnings and errors are automatically sent to the browser as
  [Toast Notifications](#-comfyui-toast-notifications), so users don't have to check the console for critical issues.
- **Debug Control:** Integrates with ComfyUI's `--verbose` flag and allows for per-node debugging via an environment variable
  (`{NODES_NAME}_NODES_DEBUG=1`).

To use the logger, in the `/__init__.py` use as the first import:

```python
from .src.nodes import nodes, main_logger
```

This will pull the `/src/nodes/__init__.py` which should include:

```python
from seconohe.logger import initialize_logger

NODES_NAME = "NameForTheNodes"
main_logger = initialize_logger(NODES_NAME)
```

NODES_NAME will be used in the logs (`[{NODES_NAME}] ...`), and will be used for the environment variable name (`{NODES_NAME}_NODES_DEBUG` all uppercase)

In your `/src/nodes/nodes.py` you get the logger like this (the main one):

```python
from . import main_logger
logger = main_logger
```

And then use it as any logger from `logging`, i.e. `logger.error("An error")`

In your `/src/nodes/utils/yyyyy.py` code you can import the `main_logger` or you can create a local one:

```python
from .. import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.yyyyy")
```

As this logger starts with NODES_NAME it will inherit all the `main_logger` goodies.


### &#x0001F35E; ComfyUI Toast Notifications

Asking users to look at ComfyUI console logs is ridiculous.
IMHO any node trying to really notify the user must do it in the browser.
And this is not available from the Python side as a standard mechanism.

I already commented it on Discord and got some attention from ComfyUI people
([see this RFC](https://github.com/Comfy-Org/rfcs/discussions/34)), so I guess this problem will be solved in the future.

Currently the only way to achieve it is using Java Script.

If you register the SeCoNoHe JS code you get a service for it.
Note that currently I don't have a simple mechanism to add SeCoNoHe scripts to JS scripts in your own node.
You might copy the files to your local JS directory.

If you want to simply register SeCoNoHe extensions do it:

In your `/__init__.py` add:

```python
from seconohe import JS_PATH
WEB_DIRECTORY = JS_PATH
```

If you do it the `logger.warning` and `logger.error` messages will be logged to the console and also sent to the browser.

If you want to send a custom message import this:

```python
from seconohe.comfy_notification import send_toast_notification
```

Here is the current protype:

```python
def send_toast_notification(logger: logging.Logger, message: str, summary: str = "Warning", severity: str = "warn",
                            sid: Optional[str] = None):
    """
    Sends a toast notification event to the ComfyUI client.

    Args:
        logger (logging.Logger): The logger used in case we need to report an error
        message (str): The message content of the toast.
        severity (str): The type of toast. Can be 'success' | 'info' | 'warn' | 'error' | 'secondary' | 'contrast'
        summary (str): Short explanation
        sid (str, optional): The session ID of the client to send to.
                            If None, broadcasts to all clients. Defaults to None.
    """
```


### &#x0001F4BE; File Downloader

The objectives are:

- Notify the user that we are downloading a file
- Show progress in the console **and** the browser
- Notify the user about successful download, or an error
- Clearly log the file origin and destination

Surprisingly I never saw a node implementing all these objectives for its download.
To get the start and end notifications you must register the [Toast Notifications](#-comfyui-toast-notifications).

To use the downloader import:

```python
from seconohe.downloader import download_file
```

And then use this function:

```python
def download_file(logger: logging.Logger, url: str, save_dir: str, file_name: str, force_urllib: bool = False,
                  kind: str = "model"):
    """
    Downloads a file from a URL with progress bars for both console and ComfyUI.
    Also GUI notification at start and end.
    We also log the URL and destination to the console.

    Args:
        logger (logging.Logger): The used logger
        url (str): The direct download URL for the file.
        save_dir (str): The directory where the file will be saved.
        file_name (str): The name of the file to be saved on disk.
        force_urllib (bool=False): Ignore `requests`
        kind (str='model'): Kind of file we are downloading, just for the logs
    """
```


### &#x270D;&#xFE0F; Automatic Node Registration

Manually maintaining the `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py` is tedious and error-prone.
This helper automates the process.

You just need to add a couple of extra members to your classes:

- `UNIQUE_NAME`: The unique name that identifies the node
- `DISPLAY_NAME`: The name the user will see in the browser

Here is an example:

```python
    class ImageDownload:
        FUNCTION = "load_or_download_image"
        CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
        DESCRIPTION = ("Downloads an image to ComfyUI's 'input' directory if it doesn't exist, then loads it using the "
                       "built-in LoadImage logic.")
        UNIQUE_NAME = "SET_ImageDownload"
        DISPLAY_NAME = "Image Download and Load"
```

Once all your nodes has these two extra members you use the following code in `/__init__.py`:

```python
from .src.nodes import nodes_xxxx, nodes_yyyy
from seconohe.register_nodes import register_nodes


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes(main_logger, [nodes_xxxx, nodes_yyyy])
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

```

Of course you can register nodes from just one file:

```python
from .src.nodes import my_nodes
from seconohe.register_nodes import register_nodes


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes(main_logger, [my_nodes])
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

```

This is all you need.

If you declare a variable named `SUFFIX` in your module all the display names for the nodes in the module will have " {SUFFIX}" added.


### &#x2699;&#xFE0F; PyTorch Helpers

`get_torch_device_options`: returns a list of devices suitable for a combo so the user can choose the inference device.
It also returns a suitable default value. Example:

```python
 from seconohe.torch import get_torch_device_options

 ...

    @classmethod
    def INPUT_TYPES(cls):
        device_options, default_device = get_torch_device_options()
        return {
            "required": {
                "target_device": (device_options, {
                    "default": default_device,
                    "tooltip": "The device (CPU or CUDA) to which the projection layer will be assigned for computation."}),
            }
        }
```

`get_offload_device`: returns a torch device to move the model after use. Is basically a wrapper for
`comfy.model_management.unet_offload_device()`, but can be used from a tool even when no ComfyUI is available.

`get_canonical_device`: return a canonical name for a torch device. This is useful to compare torch devices, so we don't think that
`cuda` and `cuda:0` are different.

`model_to_target` context: this context can be used to wrap the inference of a model, it:
- Moves the model to its designated `model.target_device`.
- Sets `torch.backends.cudnn.benchmark` based on `model.cudnn_benchmark_setting` if available.
- Sets the model to `eval()` mode.
- Wraps the operation in a `torch.no_grad()` context.
- Offloads the model to the CPU (`mm.unet_offload_device()`) afterwards.

```python
                model.target_device = self.device
                logger.debug("Using PyTorch Audio chunking for old model")
                with model_to_target(logger, model):
                    separated_tensors = separate_sources(model, input_tensor_on_device)
```

Note that this tries to offload the model even if the inference fails. Avoiding the classic VRAM waste after a fail.

`get_pytorch_memory_usage_str` is used to get a string with the memory usage.

`TorchProfile` can be used to measure the time and VRAM consumed by a CUDA task.


### &#x0001F39B;&#xFE0F; Changing Widget Values

If during the execution of a node you need to change the value assigned to a widget of the node you can use it.

```
from seconohe.comfy_node_action import send_node_action

...

send_node_action(logger, "change_widget", WIDGET_NAME, NEW_VALUE)
```

Note that you must register the JS extensions like with the [Toast Notifications](#-comfyui-toast-notifications).


### Batch iterator

Batch processing can speed up many operations, but you can't handle arbitrary batch sizes.
In ComfyUI a video is just a batch of images (B, H, W, 3).
So a node that gets a batch as input and tries to process the whole batch at once will most probably go OOM when you
connect a long video at its input.

Most nodes avoids this problem by just iterating on the batch:

```python
for image in images:
    ....
```

But in this case you just remove any advantage of processing a batch in parallel.
The best solution is to add a `batch_size` parameter and then process no more than `batch_size` at once.
So you can run in parallel, but with a limit.

Implementing it isn't complex, but makes the code less clear. For this reason SeCoNoHe provides a class to abstract it.
You use it like this:

```python
from seconohe.bti import BatchedTensorIterator
batched_iterator = BatchedTensorIterator(
    tensor=large_batch,
    sub_batch_size=SUB_BATCH_SIZE,
    device=TARGET_DEVICE
)
for i, batch_range in enumerate(batched_iterator):
    sub_batch = batched_iterator.get_batch(batch_range)

    # Use the slice here

    del sub_batch  # Avoid two instances at the same time in the target device
```

You don't need the `del` if you don't care about having two slices alloocated at the same time during a short period of time.
You might also do:

```python
for i, batch_range in enumerate(batched_iterator):
    do_something(batched_iterator.get_batch(batch_range))
```

Taking advantage of the implicit deletion of the temporal slice.
If you can't afford having two of the slices in the target device at the same time you might want to use:

```python
for i, batch_range in enumerate(batched_iterator):
    sub_batch = None  # Ensure variable exists before the try block
    try:
        # 1. Acquire the resource
        sub_batch = batched_iterator.get_batch(batch_range)

        # 2. Use the resource
        # Your processing code goes here.
        # If an error happens, the `finally` block is still executed.

    finally:
        # 3. Guarantee the resource is released
        if sub_batch is not None:
            del sub_batch
```

If you ask why don't just use `TensorDataset` and `DataLoader`: in my experience it creates a temporal copy of the whole tensor.


### Color parser

To convert a color string into its components:

- `color_to_rgb_uint8` the returned components are integers in the [0, 255] range
- `color_to_rgb_float` the returned components are floats in the [0, 1] range

The color can be represented using:

- An hexadecimal RGB value: `#RRGGBB` or `RRGGBB`, like in web applications, i.e. `#AB8020`
- Comma separated integers: `r, g, b`, i.e. `171,128,32`
- Comma separated floats: `r, g, b`, i.e. `0.67,0.5,0.13`
- Some 20 color names

When PIL is installed, should be always for ComfyUI, also supports:

- Short hexa values: `#RGB` i.e. `#B82`
- Explicit `rgb` keyword: `rgb(R,G,B)` i.e. `rgb(171,128,32)`
- Percentages: `rgb(R%,G%,B%)` i.e. `rgb(67%,50%,13%)`
- Hue-Saturation-Lightness (HSL): `hsl(H,S%,L%)`
- Hue-Saturation-Value (HSV): `hsv(H,S%,V%)`
- Around 140 color names

Features:

- Spaces at the beginning, end or around the commas are supported
- Uppercase and lowercase hexadecimals
- A single value is promoted to a gray scale, i.e. 128 -> `128,128,128`
- If blue is 0 you can just use `0.67,0.5`


### Foreground estimation

When using background removal models we get a mask that can help to separate the foreground from the background.
The problem is that when we change th background we get part of the old background at the edges.
This problem is well explained in the [Fast Multi-Level Foreground Estimation](https://arxiv.org/abs/2006.14970) paper.

In order to replace the background we need to estimate the actual foreground. We implement two algorithms:

1. Fast Multi-Level Foreground Estimation: from [PyMatting](https://github.com/pymatting/pymatting). Is very good, but somehow slow.
   The authors uses [Numba](https://numba.pydata.org/) to accelerate it. This is an excellent solution. SeCoNoHe also adds a PyTorch
   approximation to the algorithm. So you don't even need Numba installed.
   A wrapper is provided to select the best available backend.
   See `seconohe.foreground_estimation.fmlfe`
2. [Approximate Fast Foreground Colour Estimation](https://github.com/Photoroom/fast-foreground-estimation). A really fast method.
   The original code uses OpenCV, here we have a full PyTorch implementation. See `seconohe.foreground_estimation.affce`

The `appy_mask.py` provides a wrapper to AFFCE to implement a node. It can add a solid-color background or just created a refined
alpha channel.


## &#x0001F680; Examples of Nodes Using SeCoNoHe

- [Image Misc](https://github.com/set-soft/ComfyUI-ImageMisc)
- [Audio Separation](https://github.com/set-soft/AudioSeparation)
- [Audio Batch](https://github.com/set-soft/ComfyUI-AudioBatch)


## &#x0001F4DC; Project History

- 1.0.0 2025-07-24: Initial release.
- 1.0.1 2025-07-25: Better typing hints and docs
- 1.0.2 2025-07-26: Optional version info when registering the nodes
- 1.0.3 2025-10-05:
   - Better download: resume, less updates in the GUI, declare a common user agent to avoid 403 errors
   - Added "Fast Multi-Level Foreground Estimation" (from [PyMatting](https://github.com/pymatting/pymatting)) and "Approximate Fast Foreground Colour Estimation" (from [Photoroom](https://github.com/Photoroom/fast-foreground-estimation))
   - Added mechanism to apply a mask to an image
   - Added PyTorch memory usage
   - Added code to convert a color in various formats to a tuple
- 1.0.4 2025-10-22:
   - Better color parsing, now supports all PIL formats (rgb, hsv, hsl rgb%)
   - Added CUDA profiler (timer and peak memory use)
   - Apply mask: allow replacing by an image
   - Added batch iterator
- 1.0.5 2025-10-23:
   - Fixed missing colors on logger

## &#x2696;&#xFE0F; License

[GPL-3.0](LICENSE)


## &#x0001F64F; Attributions

- Main author: [Salvador E. Tropea](https://github.com/set-soft)
- Assisted by Gemini 2.5 Pro
