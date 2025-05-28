import FilePatcher

root = "xDiT/xfuser"
target_type = "*.py"

patches_wildcard = [
    {
        "replace": [
            {
                "from": "from xfuser.",
                "to":   "from xDiT.xfuser.",
            },
        ],
    },
    {
        "replace": [
            {
                "from": "from xfuser ",
                "to":   "from xDiT.xfuser ",
            },
        ],
    },
    {
        "replace": [
            {
                "from": "import xfuser.",
                "to":   "import xDiT.xfuser.",
            },
        ],
    },
    {
        "replace": [
            {
                "from": "import xfuser ",
                "to":   "import xDiT.xfuser ",
            },
        ],
    },
]

FilePatcher.patch_wildcard(root, target_type, patches_wildcard)

