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

patches = [
    {
        "file_name": f"{root}/core/long_ctx_attention/ring/ring_flash_attn.py",
        "replace": [
            # Quick & dirty fix for disabling flash_attn when it isn't supported
            # maybe I'll do a PR later, maybe I'll forget, but for now it just works.
            {
                "from": """
            block_out, block_lse = fn(
                q,
                key,
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )""",
                "to": """
            global flash_attn
            block_out, block_lse = fn(
                q,
                key,
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
                op_type="efficient" if flash_attn is None else "flash",
            )"""
            }
        ]
    },
    {
        "file_name": f"{root}/core/distributed/parallel_state.py",
        "replace": [
            # Increase torch timeout to 1 day
            {
                "from": """torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )""",
                "to":   """from datetime import timedelta
        torch.distributed.init_process_group(
            timeout=timedelta(days=1),
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )""",
            },
        ],
    },
]

FilePatcher.patch_wildcard(root, target_type, patches_wildcard)
FilePatcher.patch(patches)
