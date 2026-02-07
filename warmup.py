import sys
import traceback

from rembg import new_session

MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
    "bria-rmbg",
]


if __name__ == "__main__":
    ok = []
    fail = []

    print("Starting rembg model warmup/download...\n")
    for m in MODELS:
        print(f"Downloading / initializing model: {m}")
        try:
            # Prefer CUDA, fall back to CPU inside rembg if needed
            _ = new_session(
                model_name=m,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            ok.append(m)
            print(f"OK: {m}\n")
        except Exception as e:
            fail.append((m, str(e)))
            print(f"FAIL: {m}")
            traceback.print_exc()
            print()

    print("\n=== SUMMARY ===")
    print("OK:", ok)
    print("FAILED:", [x[0] for x in fail])
    if fail:
        print("\nDetails:")
        for m, err in fail:
            print(f"- {m}: {err}")

    # Exit with non-zero if any model failed
    sys.exit(0 if not fail else 1)
