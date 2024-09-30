import asyncio
import subprocess

async def run_script(script_name, *args):
    """
    Function to run the python script asynchronously.
    """
    process = await asyncio.create_subprocess_exec(
        'python', script_name, *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if stdout:
        print(f"[{script_name}]: {stdout.decode()}")
    if stderr:
        print(f"[{script_name} ERROR]: {stderr.decode()}")

    return process.returncode

async def main():
    # Launch both scripts asynchronously
    camera_id1 = 'Camera In'
    camera_id2 = 'Camera Out'
    task1 = asyncio.create_task(run_script('license_detector.py'))
    task2 = asyncio.create_task(run_script('vehicle_detector.py', camera_id1))
    task3 = asyncio.create_task(run_script('vehicle_detector.py', camera_id2))
    task4 = asyncio.create_task(run_script('manage.py', 'runserver'))

    # Wait for both scripts to complete
    await task1
    await task2
    await task3
    await task4

if __name__ == '__main__':
    asyncio.run(main())
