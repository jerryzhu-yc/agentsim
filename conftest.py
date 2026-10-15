"""pytest 根配置：把项目根目录加进 sys.path，直接可跑，无需安装包。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
