import os
import time
import cv2 as cv
from pathlib import Path
from imutils.video import VideoStream

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data_and_features" / "Data_original"
    data_path.mkdir(parents=True, exist_ok=True)

    mappings = {"i": "ignore", "k": "knife", "f": "fork", "s": "spoon"}
    frame_size = (1920, 1080)

    kMappings = {ord(k): v for k, v in mappings.items()}

    vs = VideoStream(src=1, usePiCamera=False, resolution=frame_size).start()
    time.sleep(1.0)
    print(mappings)

    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue

            cv.imshow("Frame", frame)
            k = cv.waitKey(1) & 0xFF

            if k == ord("q") or k == 27:
                break
            elif k in kMappings:
                cls_dir = data_path / kMappings[k]
                cls_dir.mkdir(parents=True, exist_ok=True)

                p = cls_dir / f"{int(time.time_ns())}.png"
                print(f"[INFO] saving frame: {p}")
                cv.imwrite(str(p), frame)
    finally:
        try:
            vs.stop()
        except Exception:
            pass
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
