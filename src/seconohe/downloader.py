# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# File downloader w/TQDM and ComfyUI progress
# Original code from Gemini 2.5 Pro
import logging
import os
# Requests is better than the core Python urllib, and is a really common package
# But we don't really need it. Lets make it optional:
try:
    import requests
    with_requests = True
except ImportError:
    with_requests = False
from urllib.error import URLError
from urllib.request import urlretrieve
from tqdm import tqdm
# ComfyUI imports
try:
    import comfy.utils
    with_comfy = True
except ImportError:
    with_comfy = False
# Local imports
from .comfy_notification import send_toast_notification


def _download_model_requests(logger: logging.Logger, url: str, save_dir: str, file_name: str) -> str:
    """
    Downloads a file using the `requests` library with streaming for progress.
    Progress is displayed to both console and ComfyUI.

    :param logger: Logger for status messages.
    :type logger: logging.Logger
    :param url: The direct download URL.
    :type url: str
    :param save_dir: The directory to save the file in.
    :type save_dir: str
    :param file_name: The name for the saved file.
    :type file_name: str
    :raises requests.exceptions.RequestException: For network-related errors.
    :raises IOError: If the downloaded file size does not match the expected size.
    :return: The full path to the downloaded file.
    :rtype: str
    """
    full_path = os.path.join(save_dir, file_name)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    try:
        # Use a streaming request to handle large files and get content length
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Get total file size from headers
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # --- Setup Progress Bars ---
            # Console progress bar using tqdm
            progress_bar_console = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {file_name}"
            )

            # ComfyUI progress bar
            progress_bar_ui = comfy.utils.ProgressBar(total_size_in_bytes) if with_comfy else None

            # --- Download Loop ---
            downloaded_size = 0
            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        chunk_size = len(chunk)

                        # Update console progress bar
                        progress_bar_console.update(chunk_size)

                        # Update ComfyUI progress bar
                        downloaded_size += chunk_size
                        if progress_bar_ui:
                            progress_bar_ui.update(chunk_size)  # ProgressBar takes absolute value, but update is incremental

                        # Write chunk to file
                        f.write(chunk)

            # --- Cleanup ---
            progress_bar_console.close()

            # Final check to see if download was complete
            if total_size_in_bytes != 0 and progress_bar_console.n != total_size_in_bytes:
                logger.error("Download failed: Size mismatch.")
                # Optional: remove partial file
                # os.remove(full_path)
                raise IOError(f"Download failed for {file_name}. Expected {total_size_in_bytes} but got "
                              f"{progress_bar_console.n}")

        return full_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while downloading {file_name}: {e}")
        # Clean up partial file if it exists
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass
        raise
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass
        raise


# A simple version implemented using the Python urllib
class _Downloader:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model_full_name = os.path.join(self.model_path, self.model_name)
        # Ensure the directory for the model_path exists before __init__ if used elsewhere
        # or create it at the start of download_model

    # A TQDM helper class for urlretrieve reporthook
    # This is a common pattern for this use case.
    class _TqdmUpTo(tqdm):
        """
        Provides `update_to(block_num, block_size, total_size)`
        and updates the TQDM bar.
        """
        def __init__(self, unit, unit_scale, unit_divisor, miniters, desc):
            super().__init__(unit=unit, unit_scale=unit_scale, unit_divisor=unit_divisor, miniters=miniters, desc=desc)
            self.ui_bar = None
            self.total = None

        def update_to(self, block_num=1, block_size=1, total_size=None):
            """
            block_num  : int, optional
                Number of blocks transferred so far [default: 1].
            block_size : int, optional
                Size of each block (in tqdm units) [default: 1].
            total_size : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if total_size is not None and self.total is None:
                self.total = total_size
                # ComfyUI progress bar
                if self.ui_bar is None and with_comfy:
                    self.ui_bar = comfy.utils.ProgressBar(total_size)
            # self.update() will take the *difference* from the last call.
            # So we pass the number of new blocks * block_size.
            # Since block_num is cumulative, we calculate the new amount.
            chunk_size = block_num * block_size - self.n
            self.update(chunk_size)  # self.n is current progress
            if self.ui_bar:
                self.ui_bar.update(chunk_size)  # ProgressBar takes absolute value, but update is incremental

    def download_model(self, url: str):
        try:
            # Ensure the directory exists
            # Use or '.' for current dir if dirname is empty
            os.makedirs(self.model_path or '.', exist_ok=True)

            # Get filename for tqdm description
            filename = self.model_name

            # Use _TqdmUpTo as a context manager
            with self._TqdmUpTo(unit='iB', unit_scale=True, unit_divisor=1024, miniters=1,
                                desc=f"Downloading {filename}") as t:
                # urlretrieve(url, filename=None, reporthook=None, data=None)
                # reporthook is called with (block_num, block_size, total_size)
                urlretrieve(url, self.model_full_name, reporthook=t.update_to)
            # The 'with' statement ensures t.close() is called.

            return filename

        except URLError as e:  # More specific exception for network issues
            # Clean up partially downloaded file if an error occurs
            if os.path.exists(self.model_full_name):
                os.remove(self.model_full_name)
            raise Exception(f"An error occurred while downloading the model (URL Error): {e.reason} from {url}")

        except Exception as e:
            # Clean up partially downloaded file if an error occurs
            if os.path.exists(self.model_full_name):
                os.remove(self.model_full_name)
            raise Exception(f"An unexpected error occurred while downloading the model: {e}")


def _download_model_urllib(url: str, save_dir: str, file_name: str):
    return _Downloader(save_dir, file_name).download_model(url)


def download_file(logger: logging.Logger, url: str, save_dir: str, file_name: str, force_urllib: bool = False,
                  kind: str = "model") -> str:
    """
    Downloads a file with progress reporting for both console and ComfyUI.

    This function acts as a high-level wrapper, preferring the `requests` library
    for its robustness but falling back to Python's built-in `urllib` if
    `requests` is not available. It also sends UI notifications at the start
    and end of the download.

    :param logger: The logger instance for status and error messages.
    :type logger: logging.Logger
    :param url: The direct download URL for the file.
    :type url: str
    :param save_dir: The directory where the file will be saved.
    :type save_dir: str
    :param file_name: The name of the file to be saved on disk.
    :type file_name: str
    :param force_urllib: If ``True``, forces the use of `urllib` even if
                         `requests` is available. Defaults to ``False``.
    :type force_urllib: bool
    :param kind: A descriptive string for the type of file being downloaded
                 (e.g., 'model', 'config'), used for logging. Defaults to 'model'.
    :type kind: str
    :raises Exception: Propagates exceptions from the underlying downloaders
                       (e.g., network errors, file errors).
    :return: The full path to the successfully downloaded file.
    :rtype: str
    """
    logger.info(f"Downloading {kind}: {file_name}")
    logger.info(f"Source URL: {url}")
    full_name = os.path.join(save_dir, file_name)
    logger.info(f"Destination: {full_name}")

    send_toast_notification(logger, f"Downloading `{file_name}`", "Download")

    if with_requests and not force_urllib:
        _download_model_requests(logger, url, save_dir, file_name)
    else:
        _download_model_urllib(url, save_dir, file_name)

    send_toast_notification(logger, "Finished downloading", "Download", 'success')

    logger.info(f"Successfully downloaded {full_name}")
    return full_name
