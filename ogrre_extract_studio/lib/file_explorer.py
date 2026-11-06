import os
import re
from pathlib import Path
from dash import html

def get_directory_items(cwd_path: Path, ext_filter: str = "all", only_dirs: bool = False):
    """
    Scans directory and returns UI items for directories and valid files.
    """
    if not cwd_path.exists() or not cwd_path.is_dir():
        return [], "Directory does not exist."

    ext_filter = ext_filter.lower().lstrip('.')
    dir_items = []
    file_items = []

    try:
        # Sort directories first, then files alphabetically
        entries = sorted(cwd_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        for entry in entries:
            rel_name = entry.name

            if entry.is_dir():
                dir_items.append(
                    html.Div(
                        children=[
                            html.Div(className="gui__fileList--icons gui__fileList--folder__icon"),
                            html.H4(rel_name, className="gui__fileList--file__text")
                        ],
                        # Removed 'is_dir': True to match Input({'type': 'fileList', 'index': ALL})
                        id={"type": "fileList", "index": rel_name},
                        className="gui__fileList--file explorer-item-dir",
                        n_clicks=0
                    )
                )
            elif entry.is_file() and not only_dirs:
                file_ext = entry.suffix.lower().lstrip('.')
                if ext_filter in ["all", "*", ""] or file_ext == ext_filter:
                    file_items.append(
                        html.Div(
                            children=[
                                html.Div(className="gui__fileList--icons gui__fileList--file__icon"),
                                html.H4(rel_name, className="gui__fileList--file__text")
                            ],
                            # Removed 'is_dir': False to match Input({'type': 'fileList', 'index': ALL})
                            id={"type": "fileList", "index": rel_name},
                            className="gui__fileList--file explorer-item-file",
                            n_clicks=0
                        )
                    )

        return dir_items + file_items, ""

    except PermissionError:
        return [], "Permission denied accessing folder."
    except Exception as e:
        return [], f"Error reading directory: {str(e)}"


def update_folder_history(history: list[str], target_cwd: str, is_back_action: bool = False) -> list[str]:
    """Maintains a clean stack of up to 10 historical paths."""
    if not history:
        return [target_cwd]

    if is_back_action:
        return history[:-1] if len(history) > 1 else history

    if history[-1] != target_cwd:
        new_history = history + [target_cwd]
        return new_history[-10:]  # Retain max 10 entries

    return history


def get_file_list(current_files, cwd, valid_extensions, only_dirs=False):
    """
    Generates a categorized list of file representations for a graphical user interface.

    This function categorizes files and directories from the specified list of current
    files into directories, regular files, and unknown types. It then creates a
    structured list of file representations suitable for display in a Graphical User
    Interface, applying distinct icons and formats for each type.

    Parameters:
    current_files (list[str]): A list of file paths to categorize and render.
    cwd (str): The current working directory, which is used to adjust file display names.
    valid_extensions (list[str]): A list of valid extensions for file paths.

    Returns:
    List[html.Div]
        A list of dash.html.Div elements, each representing a categorized file or directory.
    """
    # Ensure extensions are lowercased and dot-prefixed
    valid_extensions = [e.lower() if e.startswith('.') else f".{e.lower()}" for e in valid_extensions]

    dir_list = []
    file_list = []
    unknown_list = []
    for f in current_files:
        if (os.path.isfile(f)
                and len(valid_extensions) > 0
                and os.path.splitext(f)[1] not in valid_extensions):
            continue
        elif (os.path.isfile(f)
              and (os.path.splitext(f)[1] in valid_extensions
                   or len(valid_extensions) == 0)):
            icon_name = 'gui__fileList--file__icon'
        elif os.path.isdir(f):
            icon_name = 'gui__fileList--folder__icon'
        else:
            icon_name = 'gui__fileList--unknown__icon'

        if len(cwd) < 3:
            current_file = f[3:]
        else:
            current_file = f.split("\\")[-1]

        file_div = html.Div(
            id={'type': 'fileList', 'index': current_file},
            className='gui__fileList--file',
            children=[
                html.Div(
                    id={'type': 'fileIcon', 'index': current_file},
                    className=f"gui__fileList--icons {icon_name}",
                    children=[]
                ),
                html.H4(
                    id={'type': 'fileName', 'index': current_file},
                    className='gui__fileList--file__text',
                    children=f"{current_file}"
                )
            ],
            n_clicks=0
        )


        if os.path.isfile(f) and not only_dirs:
            file_list.append(file_div)
        elif os.path.isdir(f):
            dir_list.append(file_div)
        elif not only_dirs:
            unknown_list.append(file_div)
        else:
            pass

    file_list_sorted = sort_file_list(file_list)
    all_files = dir_list + file_list_sorted + unknown_list

    return all_files


def sort_file_list(file_list):
    def sort_key(m_file):
        filename = m_file.id["index"]
        match = re.search(r'ControlFile_ex(\d+)(.*)\.yaml', filename)
        if match:
            number = int(match.group(1))
            suffix = match.group(2)
            return (number, suffix)
        else:
            return (float("inf"), filename)

    return sorted(file_list, key=sort_key)


def check_file_creatable(file_path: Path) -> tuple[bool, str]:
    """Checks if a file path can be created or written to without leaving a file on disk."""
    parent = file_path.parent
    if not parent.exists():
        return False, f"Directory does not exist: {parent}"

    if file_path.exists():
        if file_path.is_dir():
            return False, f"'{file_path.name}' is a directory."
        if not os.access(file_path, os.W_OK):
            return False, f"File '{file_path.name}' is read-only."
    else:
        if not os.access(parent, os.W_OK):
            return False, f"Permission denied: Cannot write to directory {parent}"

        # Test creation/filename validity and immediately remove the test file
        file_created = False
        try:
            with open(file_path, 'a'):
                file_created = True
        except Exception as e:
            return False, f"Cannot create file: {str(e)}"
        finally:
            if file_created and file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass

    return True, ""