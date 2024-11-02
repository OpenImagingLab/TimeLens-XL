from os import path as osp
from os import scandir
import importlib


# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
data_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.name.startswith('params_')]
# import all the dataset modules
_data_modules = [importlib.import_module(f'params.bsergb.{file_name}') for file_name in data_filenames]
