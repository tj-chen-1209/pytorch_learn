import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/tingjia/workspace/src/ws_moveit/install/moveit_configs_utils'
