"""
EM-TDB - SFTP remote data download and Gibbs free energy fitting.

Extends GTFitter with SFTP connectivity via paramiko.
"""

import json
import logging
import os
import re
import stat
import tempfile
import traceback
from pathlib import Path
from typing import Any

from emtdb.gibbsfit import GTFitter

# Optional paramiko import — checked lazily in connect()
try:
    import paramiko

    _HAS_PARAMIKO = True
except ImportError:
    paramiko = None  # type: ignore[assignment]
    _HAS_PARAMIKO = False

log = logging.getLogger(__name__)

# Regex to detect "$ENV_VAR" patterns in config values
_ENV_VAR_RE = re.compile(r"^\$([A-Z_][A-Z0-9_]*)$")


def _resolve_env(value: str | None) -> str | None:
    """If *value* looks like ``$VAR_NAME``, resolve it from the environment.

    Returns the original value unchanged if it doesn't match the pattern.
    """
    if not value:
        return None
    m = _ENV_VAR_RE.match(value)
    if m:
        return os.environ.get(m.group(1))
    return value


def load_sftp_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate an SFTP configuration from a JSON file.

    Expected JSON structure::

        {
          "tdb_name": "sftp_remote_fit",
          "output": "sftp_remote_fit.tdb",
          "local_dir": null,
          "targets": [
            {
              "host": "10.144.144.11",
              "port": 22,
              "username": "mcmf507",
              "password": "****",
              "key_filename": null,
              "remote_dir": "/data/EndMembers",
              "data_type": "json"
            }
          ]
        }

    ``password`` and ``key_filename`` support ``$ENV_VAR`` syntax::

        "password": "$SFTP_PASSWORD"

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A dict with keys ``targets``, and optionally ``tdb_name``,
        ``output``, ``local_dir``.

    Raises:
        FileNotFoundError: The config file does not exist.
        json.JSONDecodeError: The file contains invalid JSON.
        ValueError: Validation failed (missing fields, empty targets, …).
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)

    targets = cfg.get("targets", [])
    if not targets:
        raise ValueError("Config must contain at least one target in 'targets'")
    if not isinstance(targets, list):
        raise ValueError("'targets' must be a list")

    required = {"host", "username", "remote_dir", "data_type"}
    for i, tgt in enumerate(targets):
        missing = required - set(tgt.keys())
        if missing:
            raise ValueError(
                f"Target {i} is missing required fields: {', '.join(sorted(missing))}"
            )
        tgt.setdefault("port", 22)
        # Resolve $ENV_VAR patterns
        tgt["password"] = _resolve_env(tgt.get("password"))
        tgt["key_filename"] = _resolve_env(tgt.get("key_filename"))

    cfg.setdefault("tdb_name", "sftp_fit")
    cfg.setdefault("output", "")
    cfg.setdefault("local_dir", None)
    return cfg


class SFTPFit(GTFitter):
    """Fit Gibbs-Temperature data downloaded via SFTP.

    Usage::

        fitter = SFTPFit(PHASE_METRICS)
        results = fitter.process_sftp(
            remote_dir="/data/EndMembers",
            data_type="json",
            host="10.0.0.1",
            username="user",
            password="secret",
        )
        parsed = fitter.fit2db(results, "my_tdb")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssh_client = None
        self.sftp_client = None

    # ── connection management ────────────────────────────────────────────

    def connect(
        self,
        host: str,
        port: int = 22,
        username: str | None = None,
        password: str | None = None,
        key_filename: str | None = None,
        timeout: int = 30,
    ) -> None:
        """Connect to an SFTP server.

        Args:
            host: Server hostname or IP.
            port: SSH port.
            username: Login username.
            password: Login password (mutually exclusive with ``key_filename``).
            key_filename: Path to an SSH private key.
            timeout: Connection timeout in seconds.

        Raises:
            ImportError: ``paramiko`` is not installed.
            ConnectionError: On any authentication, network, or timeout failure.
        """
        if not _HAS_PARAMIKO:
            raise ImportError(
                "paramiko is required for SFTP support. "
                "Install it with: pip install 'em-tdb[sftp]'"
            )

        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                host,
                port=port,
                username=username,
                password=password,
                key_filename=key_filename,
                timeout=timeout,
            )
            self.sftp_client = self.ssh_client.open_sftp()
            log.info("Connected to %s:%d", host, port)
        except paramiko.AuthenticationException as e:
            raise ConnectionError(
                f"Authentication failed for {username}@{host}:{port}"
            ) from e
        except paramiko.SSHException as e:
            raise ConnectionError(f"SSH negotiation with {host}:{port} failed") from e
        except OSError as e:
            raise ConnectionError(
                f"Network error connecting to {host}:{port}: {e}"
            ) from e

    def disconnect(self) -> None:
        """Disconnect from the SFTP server.  Safe to call multiple times."""
        if self.sftp_client is not None:
            try:
                self.sftp_client.close()
            except Exception:
                pass
            self.sftp_client = None
        if self.ssh_client is not None:
            try:
                self.ssh_client.close()
            except Exception:
                pass
            self.ssh_client = None
        log.info("Disconnected from SFTP server")

    # ── download logic ───────────────────────────────────────────────────

    @staticmethod
    def _format_remote_path(path: str | Path) -> str:
        """Normalize a path to use forward slashes."""
        return str(path).replace("\\", "/")

    def download_remote_directory(
        self,
        remote_path: str | Path,
        local_path: str | Path,
        data_type: str = "json",
        recursive: bool = True,
    ) -> tuple[int, int, int]:
        """Recursively download data files from a remote SFTP directory.

        Args:
            remote_path: Remote directory path on the SFTP server.
            local_path: Local directory to download into.
            data_type: ``"json"`` or ``"dat"`` — determines which file
                extensions to download.
            recursive: Whether to descend into subdirectories.

        Returns:
            A tuple ``(downloaded, skipped, failed)``.
        """
        import paramiko

        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        extensions = {".json"} if data_type == "json" else {".dat"}
        remote_str = self._format_remote_path(remote_path)

        log.info(
            "Downloading %s → %s  [%s files]", remote_str, local_path, data_type
        )

        downloaded = 0
        skipped = 0
        failed = 0

        for item in self.sftp_client.listdir_attr(remote_str):
            name = item.filename
            is_dir = stat.S_ISDIR(item.st_mode)

            if is_dir and recursive:
                sub_local = local_path / name
                sub_local.mkdir(parents=True, exist_ok=True)
                d, s, f = self.download_remote_directory(
                    remote_path=f"{remote_str}/{name}",
                    local_path=sub_local,
                    data_type=data_type,
                    recursive=False,
                )
                downloaded += d
                skipped += s
                failed += f
                continue

            if is_dir:
                continue  # skip subdirs when not recursive

            ext = Path(name).suffix.lower()
            if ext not in extensions:
                log.debug("Skip %s (not a %s file)", name, data_type)
                skipped += 1
                continue

            try:
                remote_file = f"{remote_str}/{name}"
                local_file = local_path / name
                self.sftp_client.get(remote_file, str(local_file))
                log.info("Downloaded  %s", name)
                downloaded += 1
            except Exception as e:
                log.error("Failed to download %s: %s", name, e)
                failed += 1

        log.info(
            "Downloaded %d, skipped %d, failed %d from %s",
            downloaded,
            skipped,
            failed,
            remote_str,
        )
        return downloaded, skipped, failed

    # ── end-to-end workflow ──────────────────────────────────────────────

    def process_sftp(
        self,
        remote_dir: str | Path,
        data_type: str = "json",
        host: str | None = None,
        port: int = 22,
        username: str | None = None,
        password: str | None = None,
        key_filename: str | None = None,
        local_dir: str | Path | None = None,
        timeout: int = 30,
    ) -> list[dict[str, Any]]:
        """Download remote data via SFTP and run the fitting pipeline.

        Args:
            remote_dir: Remote directory to download.
            data_type: ``"json"`` or ``"dat"``.
            host: SFTP server hostname/IP.
            port: SSH port.
            username: Login username.
            password: Login password.
            key_filename: SSH private key path.
            local_dir: Local download directory.  If ``None`` a temporary
                directory is created (and cleaned up automatically).
            timeout: Connection timeout in seconds.

        Returns:
            List of :class:`~emtdb.gibbsfit.FitResult` dicts.
        """
        remote_str = self._format_remote_path(remote_dir)
        results: list[dict[str, Any]] = []
        _tmp_dir_owner = local_dir is None  # did we create the temp dir?

        try:
            self.connect(host, port, username, password, key_filename, timeout)

            if local_dir is None:
                local_dir = Path(tempfile.mkdtemp(prefix="sftp_"))
            else:
                local_dir = Path(local_dir)

            # Download
            self.download_remote_directory(remote_str, local_dir, data_type)

            # Fit
            results = self.process_folders(local_dir, data_type)
            log.info("Fitted %d entries from %s", len(results), remote_str)

        except Exception:
            log.error("SFTP fit failed for %s", remote_str)
            log.error(traceback.format_exc())
            raise
        finally:
            self.disconnect()
            if _tmp_dir_owner and local_dir is not None and local_dir.exists():
                import shutil

                shutil.rmtree(local_dir, ignore_errors=True)

        return results
