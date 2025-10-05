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
from urllib import request
from tqdm import tqdm
# ComfyUI imports
try:
    import comfy.utils
    with_comfy = True
    from .comfy_notification import send_toast_notification
except ImportError:
    with_comfy = False

    def send_toast_notification(logger, msg, kind, extra=''):
        pass
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
              ' Chrome/58.0.3029.110 Safari/537.36')


def _download_model_requests(logger: logging.Logger, url: str, save_dir: str, file_name: str) -> str:
    """
    Downloads a file using the `requests` library with streaming for progress
    and support for resuming partial downloads.
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
    full_path_partial = full_path + '.partial'

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # --- Check for existing partial download ---
    downloaded_size = 0
    if os.path.exists(full_path_partial):
        downloaded_size = os.path.getsize(full_path_partial)

    try:
        # --- Prepare Request ---
        headers = {'User-Agent': USER_AGENT}  # Define headers to mimic a browser
        if downloaded_size > 0:
            headers['Range'] = f'bytes={downloaded_size}-'
            logger.info(f"Resuming download for {file_name} from {downloaded_size} bytes.")

        # Use a streaming request to handle large files and get content length
        with requests.get(url, stream=True, timeout=10, headers=headers) as r:
            # --- Handle Server Response for Resuming ---
            open_mode = 'wb'  # Default to writing a new file
            is_resuming = False
            if r.status_code == 206:  # Partial Content
                open_mode = 'ab'  # Append to existing file
                is_resuming = True
                # Get total size from Content-Range header (e.g., "bytes 123-456/789")
                content_range = r.headers.get('content-range')
                if content_range:
                    total_size_in_bytes = int(content_range.split('/')[-1])
                else:
                    # Fallback if Content-Range is missing, though unlikely for 206
                    total_size_in_bytes = downloaded_size + int(r.headers.get('content-length', 0))
            else:
                r.raise_for_status()  # Raise exception for other bad statuses (4xx or 5xx)
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                if downloaded_size > 0:
                    logger.warning(f"Server did not support resume for {file_name}. Starting download from beginning.")
                    downloaded_size = 0  # Reset if server sends the full file

            # --- Setup Progress Bars ---
            # Download 200 blocks at most, ComfyUI 0.3.57 does a sync draw so it gets slow if we ask for thousands of updates
            block_size = max(total_size_in_bytes // 200, 65536)

            # Console progress bar using tqdm
            progress_bar_console = tqdm(
                total=total_size_in_bytes,
                unit='iB',
                unit_scale=True,
                initial=downloaded_size,  # Set the initial progress
                desc=f"Downloading {file_name}"
            )

            # ComfyUI progress bar
            progress_bar_ui = comfy.utils.ProgressBar(total_size_in_bytes) if with_comfy else None
            if progress_bar_ui and is_resuming:
                # Manually update ComfyUI bar to its starting point
                progress_bar_ui.update(downloaded_size)

            # --- Download Loop ---
            with open(full_path_partial, open_mode) as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        chunk_len = len(chunk)
                        progress_bar_console.update(chunk_len)
                        if progress_bar_ui:
                            progress_bar_ui.update(chunk_len)
                        f.write(chunk)

            # --- Cleanup ---
            progress_bar_console.close()

            # --- Final Verification ---
            # Use final file size on disk for verification
            final_size = os.path.getsize(full_path_partial)
            if total_size_in_bytes != 0 and final_size != total_size_in_bytes:
                logger.error("Download failed: Size mismatch.")
                raise IOError(f"Download failed for {file_name}. Expected {total_size_in_bytes} but got "
                              f"{final_size}")

            os.rename(full_path_partial, full_path)

        return full_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while downloading {file_name}: {e}. Partial file preserved for future resume.")
        # DO NOT remove the partial file, allow resuming later
        raise
    except Exception as e:
        logger.error(f"An error occurred during download: {e}. Partial file preserved for future resume.")
        # DO NOT remove the partial file, allow resuming later
        raise


# A simple version implemented using the Python urllib
class _Downloader:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model_full_name = os.path.join(self.model_path, self.model_name)
        self.model_full_name_partial = self.model_full_name + '.partial'
        # Ensure the directory for the model_path exists before __init__ if used elsewhere
        # or create it at the start of download_model

    def download_model(self, url: str):
        try:
            # Ensure the directory exists
            os.makedirs(self.model_path or '.', exist_ok=True)

            # 1. Create a Request object with custom headers
            headers = {'User-Agent': USER_AGENT}
            req = request.Request(url, headers=headers)

            # 2. Open the URL and get the response
            with request.urlopen(req) as response:
                # Check if the server sent a Content-Length header
                total_size = int(response.headers.get('Content-Length', 0))
                filename = self.model_name

                # 3. Manually download the file in chunks with tqdm progress bar
                with open(self.model_full_name_partial, 'wb') as f_out, tqdm(
                    desc=f"Downloading {filename}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    ui_bar = None
                    if with_comfy:
                        ui_bar = comfy.utils.ProgressBar(total_size)

                    # Read and write in chunks
                    chunk_size = max(total_size // 200, 65536)
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        bar.update(len(chunk))
                        if ui_bar:
                            ui_bar.update(len(chunk))

            # Rename the partial file to the final filename upon success
            os.rename(self.model_full_name_partial, self.model_full_name)

            return filename

        except URLError as e:  # More specific exception for network issues
            # Clean up partially downloaded file if an error occurs
            if os.path.exists(self.model_full_name_partial):
                os.remove(self.model_full_name_partial)
            raise Exception(f"An error occurred while downloading the model (URL Error): {e.reason} from {url}")

        except Exception as e:
            # Clean up partially downloaded file if an error occurs
            if os.path.exists(self.model_full_name_partial):
                os.remove(self.model_full_name_partial)
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


if __name__ == '__main__':
    if False:
        download_file(logging.getLogger(__name__), 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/'
                      'Balearica_regulorum_1_Luc_Viatour.jpg/1080px-Balearica_regulorum_1_Luc_Viatour.jpg',
                      '.', 'test.jpg', force_urllib=False, kind="image")
    if True:
        download_file(logging.getLogger(__name__), 'https://huggingface.co/ZhengPeng7/BiRefNet_HR/resolve/main/'
                      'model.safetensors', '.', 'General-HR.safetensors', force_urllib=False, kind="model")
