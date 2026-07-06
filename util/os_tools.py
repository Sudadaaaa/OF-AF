import glob
import os


def make_glob_filename_iterator(file_template):
    return sorted(glob.glob(file_template))


def find_leaf_folders(root_folder):
    """
        返回Leaf文件夹列表。
        “Leaf”文件夹是没有非隐藏子文件夹的非隐藏文件夹。
    """
    leaf_folders = []
    for sequence_folder, child_folders, _ in os.walk(root_folder):
        # 跳过隐藏目录。
        if os.path.basename(sequence_folder)[0] == ".":
            continue
        # 删除隐藏目录。
        child_folders = [
            child_folder for child_folder in child_folders if child_folder[0] != "."
        ]
        if not child_folders:
            leaf_folders.append(sequence_folder)
    return leaf_folders


def find_files_by_template(folder, file_template, is_recursive=False):
    """
        返回与模板匹配的文件列表。搜索在所有子文件夹中递归执行。
    """
    if is_recursive:
        return sorted(glob.glob(os.path.join(folder, '**', file_template), recursive=True))
    return sorted(glob.glob(os.path.join(folder, file_template)))


def make_filename_iterator(filename_template):
    """
        返回文件名上的迭代器。
        Args:
            filename_template: full path to the folder and filename template,
                               e.g. '/path/to/file/{:d}.npz'.
    """
    index = 0
    filename = filename_template.format(index)
    while os.path.isfile(filename):
        yield filename
        index += 1
        filename = filename_template.format(index)


def list_to_file(filename, lst):
    """
        将列表中的每一项保存为文件中的一行。
    """
    lst = [item.rstrip("\n") + "\n" for item in lst[:-1]] + [lst[-1].rstrip("\n")]
    with open(filename, "w") as f:
        f.writelines(lst)


def file_to_list(filename):
    """
        将文件的每一行添加到列表中。
    """
    with open(filename) as f:
        examples_list = [line.rstrip("\n") for line in f.readlines()]
    return examples_list
