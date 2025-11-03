from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class SystemVolume:
    """Control system master volume using PyCAW."""

    @staticmethod
    def set(volume_percent: int):
        """Set master volume (0-100)."""
        volume_percent = max(0, min(volume_percent, 100))
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(volume_percent / 100.0, None)

    @staticmethod
    def get() -> int:
        """Return current master volume as integer percent (0-100)."""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()
        return int(current * 100)


if __name__ == "__main__":
    SystemVolume.set(99) 
    print("Current Volume:", SystemVolume.get(), "%")
