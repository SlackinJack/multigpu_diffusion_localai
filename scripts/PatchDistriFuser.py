import FilePatcher

root = "DistriFuser/distrifuser"
models = f"{root}/models"
modules = f"{root}/modules"

patches = [
    # models
    {
        "file_name": f"{models}/base_model.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    {
        "file_name": f"{models}/distri_sdxl_unet_pp.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    {
        "file_name": f"{models}/distri_sdxl_unet_tp.py",
        "replace": [
            {
                "from": "from distrifuser.modules",
                "to":   "from ..modules",
            },
        ],
    },
    # modules/pp
    {
        "file_name": f"{modules}/pp/attn.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/pp/conv2d.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/pp/groupnorm.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    # modules/tp
    {
        "file_name": f"{modules}/tp/attention.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    {
        "file_name": f"{modules}/tp/conv2d.py",
        "replace": [
            {
                "from": "from distrifuser.modules.base_module",
                "to":   "from ..base_module",
            },
            {
                "from": "from distrifuser.utils",
                "to":   "from ...utils",
            },
        ],
    },
    # modules
    {
        "file_name": f"{modules}/base_module.py",
        "replace": [
            {
                "from": "from distrifuser.utils",
                "to":   "from ..utils",
            },
        ],
    },
]

FilePatcher.patch(patches)

