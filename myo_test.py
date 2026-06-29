import asyncio
import myo
from myo import ClassifierMode, EMGMode, IMUMode


class Listener(myo.MyoClient):
    async def on_emg_data(self, emg: myo.EMGData):
        print("EMG data:", emg.sample1, emg.sample2)

    async def on_imu_data(self, imu: myo.IMUData):
        print("IMU data:", imu)

    async def on_classifier_event(self, ce: myo.ClassifierEvent):
        print("Classifier event:", ce)

    async def on_aggregated_data(self, _ad: myo.AggregatedData):
        pass

    async def on_emg_data_aggregated(self, _eds: myo.EMGDataSingle):
        pass

    async def on_fv_data(self, _fvd: myo.FVData):
        pass

    async def on_motion_event(self, _me: myo.MotionEvent):
        pass


async def main():
    print("Scanning for Myo...")
    client = await Listener.with_device()
    print(f"Connected to {client.device.name} ({client.device.address})")

    await client.setup(
        classifier_mode=ClassifierMode.ENABLED,
        emg_mode=EMGMode.SEND_EMG,
        imu_mode=IMUMode.SEND_DATA,
    )
    await client.start()
    print("Streaming — press Enter to quit.")
    try:
        await asyncio.get_event_loop().run_in_executor(None, input)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await client.stop()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
